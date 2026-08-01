use std::{path::Path, process::Command};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceFocusPoint {
    pub time: f64,
    pub focus_x: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FaceFocus {
    pub focus_x: Option<f64>,
    pub needs_scene_fit: bool,
    pub samples: usize,
    pub faces_seen: usize,
    pub track: Vec<FaceFocusPoint>,
}

pub fn detect_face_focus(
    source_path: &str,
    start_sec: f64,
    end_sec: f64,
    work_dir: &Path,
) -> Result<Option<FaceFocus>> {
    let python = vision_python_command(work_dir);
    if !vision_dependencies_available(&python) {
        return Ok(None);
    }

    std::fs::create_dir_all(work_dir).context("creating vision helper directory")?;
    let script_path = work_dir.join("face_focus.py");
    std::fs::write(&script_path, FACE_FOCUS_SCRIPT).context("writing face focus helper")?;

    let output = Command::new(python)
        .arg(&script_path)
        .arg(source_path)
        .arg(format!("{start_sec:.3}"))
        .arg(format!("{end_sec:.3}"))
        .output()
        .context("running local face detector")?;

    if !output.status.success() {
        return Err(anyhow!(
            "local face detector failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let result: FaceFocus = serde_json::from_slice(&output.stdout)
        .context("parsing local face detector output")?;
    Ok(result.focus_x.is_some().then_some(result))
}

fn vision_dependencies_available(python: &Path) -> bool {
    Command::new(python)
        .args(["-c", "import cv2, mediapipe"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn vision_python_command(work_dir: &Path) -> std::path::PathBuf {
    if cfg!(target_os = "windows") {
        let isolated_python = work_dir.join(".venv").join("Scripts").join("python.exe");
        if isolated_python.is_file() {
            return isolated_python;
        }
        std::path::PathBuf::from("py")
    } else {
        let isolated_python = work_dir.join(".venv").join("bin").join("python");
        if isolated_python.is_file() {
            return isolated_python;
        }
        std::path::PathBuf::from("python3")
    }
}

const FACE_FOCUS_SCRIPT: &str = r#"
import cv2
import json
import math
import mediapipe as mp
import statistics
import sys

source_path = sys.argv[1]
start_sec = float(sys.argv[2])
end_sec = float(sys.argv[3])

cap = cv2.VideoCapture(source_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
start_frame = max(0, int(start_sec * fps))
end_frame = min(int(end_sec * fps), int(frame_count)) if frame_count else int(end_sec * fps)
sample_every = max(1, int(round(fps / 4.0)))
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

track = []
last_center = None
faces_seen = 0
needs_scene_fit = False

with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.55) as detector:
    frame_index = start_frame
    while frame_index <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        if (frame_index - start_frame) % sample_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            detections = []
            if results.detections:
                for detection in results.detections:
                    box = detection.location_data.relative_bounding_box
                    x = max(0.0, min(1.0, box.xmin))
                    y = max(0.0, min(1.0, box.ymin))
                    w = max(0.0, min(1.0 - x, box.width))
                    h = max(0.0, min(1.0 - y, box.height))
                    if w > 0.0 and h > 0.0:
                        detections.append({"x": x + w / 2.0, "area": w * h})

            if detections:
                largest_area = max(item["area"] for item in detections)
                if len(detections) > 1:
                    significant = [item for item in detections if item["area"] >= largest_area * 0.45]
                    if len(significant) > 1:
                        spread = max(item["x"] for item in significant) - min(item["x"] for item in significant)
                        if spread > 0.42:
                            needs_scene_fit = True

                # First sample selects the dominant face. Afterwards prefer the
                # nearby face, which avoids jumping to a poster or guest.
                if last_center is None:
                    chosen = max(detections, key=lambda item: item["area"])
                else:
                    chosen = max(
                        detections,
                        key=lambda item: item["area"] * 0.7 + (1.0 - min(1.0, abs(item["x"] - last_center))) * 0.3,
                    )
                last_center = chosen["x"]
                track.append({"time": (frame_index - start_frame) / fps, "focusX": last_center})
                faces_seen += len(detections)

        frame_index += 1

cap.release()

# Smooth only adjacent samples. A cut to a new camera/speaker gets a quick,
# controlled pan instead of locking the crop to the old face for the clip.
smoothed_track = []
for point in track:
    if smoothed_track and point["time"] - smoothed_track[-1]["time"] <= 1.0:
        smooth_x = smoothed_track[-1]["focusX"] * 0.42 + point["focusX"] * 0.58
    else:
        smooth_x = point["focusX"]
    smoothed_track.append({"time": point["time"], "focusX": max(0.0, min(1.0, smooth_x))})

focus_x = None
if smoothed_track:
    # Median is retained only as a fallback when an individual crop frame
    # cannot evaluate a tracking point.
    focus_x = max(0.0, min(1.0, statistics.median(item["focusX"] for item in smoothed_track)))

print(json.dumps({
    "focusX": focus_x,
    "needsSceneFit": needs_scene_fit,
    "samples": len(smoothed_track),
    "facesSeen": faces_seen,
    "track": smoothed_track,
}))
"#;
