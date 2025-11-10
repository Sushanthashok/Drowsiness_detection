# utils/vision.py
import cv2
import numpy as np
from math import dist
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Robust eye landmarks (MediaPipe FaceMesh indices)
LEFT_CORNERS  = (33, 133)           # left/right eye corners
RIGHT_CORNERS = (362, 263)
LEFT_VERTS    = [(159, 145), (160, 144)]   # two eyelid pairs (upper, lower)
RIGHT_VERTS   = [(386, 374), (385, 380)]

def _ear_pairwise(landmarks, corners, verts, w, h):
    """Normalized eye opening: avg vertical eyelid gap / corner distance."""
    p1 = landmarks[corners[0]]; p4 = landmarks[corners[1]]
    p1 = (int(p1.x * w), int(p1.y * h))
    p4 = (int(p4.x * w), int(p4.y * h))
    horiz = dist(p1, p4) or 1.0

    vds = []
    for up_idx, low_idx in verts:
        up = landmarks[up_idx]; lo = landmarks[low_idx]
        up = (int(up.x * w), int(up.y * h))
        lo = (int(lo.x * w), int(lo.y * h))
        vds.append(dist(up, lo))
    vertical = sum(vds) / max(1, len(vds))
    return vertical / horiz

def _age(face_bgr, DeepFace):
    """Age prediction using DeepFace. Returns -1 if unavailable."""
    if DeepFace is None or face_bgr is None or face_bgr.size == 0:
        return -1
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    try:
        res = DeepFace.analyze(rgb, actions=["age"], enforce_detection=False)
        if isinstance(res, list) and len(res) > 0:
            return int(res[0].get("age", -1))
        return int(res.get("age", -1))
    except Exception:
        return -1

def process_frame(frame_bgr, face_mesh, ear_thresh=0.28, age_box_scale=1.2, DeepFace=None):
    """
    Draw per-face RED (sleeping) / GREEN (awake) boxes and age.
    Returns:
      annotated_frame,
      stats = {"total": int, "sleeping": int, "sleepers_ages": List[int]}
    """
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    total = 0; sleeping = 0; sleepers_ages = []

    if not res.multi_face_landmarks:
        return frame_bgr, {"total": 0, "sleeping": 0, "sleepers_ages": []}

    for fl in res.multi_face_landmarks:
        total += 1

        earL = _ear_pairwise(fl.landmark, LEFT_CORNERS,  LEFT_VERTS,  w, h)
        earR = _ear_pairwise(fl.landmark, RIGHT_CORNERS, RIGHT_VERTS, w, h)
        ear  = (earL + earR) / 2.0

        xs = [int(lm.x * w) for lm in fl.landmark]
        ys = [int(lm.y * h) for lm in fl.landmark]
        x1, y1, x2, y2 = max(min(xs),0), max(min(ys),0), min(max(xs),w-1), min(max(ys),h-1)

        asleep = ear < ear_thresh
        color = (0, 0, 255) if asleep else (0, 200, 0)

        # Age crop (slightly expanded)
        age_pred = -1
        if DeepFace is not None:
            bw, bh = x2 - x1, y2 - y1
            cx, cy = x1 + bw//2, y1 + bh//2
            nw, nh = int(bw*age_box_scale), int(bh*age_box_scale)
            ax1, ay1 = max(cx - nw//2, 0), max(cy - nh//2, 0)
            ax2, ay2 = min(cx + nw//2, w-1), min(cy + nh//2, h-1)
            face_crop = frame_bgr[ay1:ay2, ax1:ax2].copy()
            age_pred = _age(face_crop, DeepFace)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        tag = "SLEEPING" if asleep else "AWAKE"
        age_txt = f" | Age:{age_pred}" if age_pred >= 0 else ""
        cv2.putText(frame_bgr, f"{tag} | EAR:{ear:.2f}{age_txt}",
                    (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if asleep:
            sleeping += 1
            if age_pred >= 0:
                sleepers_ages.append(age_pred)

    return frame_bgr, {"total": total, "sleeping": sleeping, "sleepers_ages": sleepers_ages}
