use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::models::MediaProbe;

pub fn command_exists(name: &str) -> bool {
    Command::new(name).arg("-version").output().is_ok()
}

pub fn probe_media(path: &str) -> Result<MediaProbe> {
    if !command_exists("ffprobe") {
        return Err(anyhow!("ffprobe is not installed or not available on PATH"));
    }

    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            path,
        ])
        .output()
        .context("running ffprobe")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let json: Value = serde_json::from_slice(&output.stdout).context("parsing ffprobe JSON")?;
    let streams = json
        .get("streams")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let video = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("video"));
    let audio = streams
        .iter()
        .find(|stream| stream.get("codec_type").and_then(Value::as_str) == Some("audio"));

    let duration_sec = json
        .get("format")
        .and_then(|format| format.get("duration"))
        .and_then(Value::as_str)
        .and_then(|duration| duration.parse::<f64>().ok());

    Ok(MediaProbe {
        duration_sec,
        has_video: video.is_some(),
        width: video
            .and_then(|stream| stream.get("width"))
            .and_then(Value::as_i64),
        height: video
            .and_then(|stream| stream.get("height"))
            .and_then(Value::as_i64),
        video_codec: video
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        audio_codec: audio
            .and_then(|stream| stream.get("codec_name"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

pub fn extract_audio(source_path: &str, project_dir: &Path) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    std::fs::create_dir_all(project_dir)?;
    let output_path = project_dir.join("transcription_audio.wav");

    let output = Command::new("ffmpeg")
        .args(["-y", "-i", source_path, "-vn", "-ac", "1", "-ar", "16000"])
        .arg(&output_path)
        .output()
        .context("running ffmpeg audio extraction")?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg audio extraction failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    Ok(output_path)
}

pub fn render_flat_clip(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    output_path: &Path,
    drawtext_filters: Option<&str>,
    fit_scene: bool,
    focus_x: Option<f64>,
) -> Result<PathBuf> {
    if !command_exists("ffmpeg") {
        return Err(anyhow!("ffmpeg is not installed or not available on PATH"));
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let start = format!("{start_sec:.3}");
    let end = format!("{end_sec:.3}");

    let probe = probe_media(source_path).ok();
    let has_video = probe.map(|p| p.has_video).unwrap_or(false);

    // A 4090 can make export dramatically faster through NVENC. If a machine
    // advertises the encoder but cannot actually start it, retry safely with
    // software x264 instead of failing the user's clip.
    let use_nvenc = has_video && encoder_available("h264_nvenc");
    if let Err(nvenc_error) = run_render_command(
        source_path,
        &start,
        &end,
        output_path,
        drawtext_filters,
        has_video,
        fit_scene,
        focus_x,
        use_nvenc,
    ) {
        if !use_nvenc {
            return Err(nvenc_error);
        }
        run_render_command(
            source_path,
            &start,
            &end,
            output_path,
            drawtext_filters,
            has_video,
            fit_scene,
            focus_x,
            false,
        )
        .with_context(|| format!("NVENC render failed ({nvenc_error}); software fallback also failed"))?;
    }

    Ok(output_path.to_path_buf())
}

fn encoder_available(name: &str) -> bool {
    Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).contains(name))
        .unwrap_or(false)
}

fn run_render_command(
    source_path: &str,
    start: &str,
    end: &str,
    output_path: &Path,
    drawtext_filters: Option<&str>,
    has_video: bool,
    fit_scene: bool,
    focus_x: Option<f64>,
    use_nvenc: bool,
) -> Result<()> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-i", source_path, "-ss", start, "-to", end]);

    if has_video {
        let mut filter = if fit_scene {
            // Use only when the whole scene matters (multiple people, product,
            // or wide action). The foreground stays sharp; only the side fill
            // is a blurred copy of the same source video.
            "[0:v]split=2[bgsrc][fgsrc];[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=20:10[bg];[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h):shortest=1,setsar=1".to_string()
        } else {
            // Fast portrait crop. When the vision helper detects a face, move
            // the crop around that face instead of assuming source center.
            let focus_x = focus_x.unwrap_or(0.5).clamp(0.0, 1.0);
            format!(
                "crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)':x='max(0,min(iw-ow,iw*{focus_x:.6}-ow/2))':y='(ih-oh)/2',setsar=1"
            )
        };
        if let Some(drawtext) = drawtext_filters {
            if !drawtext.is_empty() {
                filter = format!("{},{}", filter, drawtext);
            }
        }
        cmd.args(["-vf", &filter]);
        if use_nvenc {
            cmd.args(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0", "-pix_fmt", "yuv420p"]);
        } else {
            cmd.args(["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]);
        }
    } else {
        cmd.arg("-vn");
    }

    cmd.args(["-c:a", "aac", "-b:a", "192k"]);
    cmd.arg(output_path);

    let output = cmd.output().context("running ffmpeg clip render")?;
    if output.status.success() {
        Ok(())
    } else {
        Err(anyhow!(
            "ffmpeg clip render failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ))
    }
}
