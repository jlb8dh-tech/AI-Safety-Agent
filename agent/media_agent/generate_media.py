import json, os
from pathlib import Path

# Always work relative to THIS file's folder (media_agent/)
BASE_DIR = Path(__file__).parent.resolve()

ASSETS_DIR = BASE_DIR / "assets"
IMG_DIR = ASSETS_DIR / "img"
VIDEO_DIR = ASSETS_DIR / "video"
MEDIA_JSON_PATH = BASE_DIR / "media.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Dummy scenes (replace later with scene_plan.json)
scenes = [
    {"scene_id": "S1_intro", "goal": "Introduce warehouse environment"},
    {"scene_id": "S2_hazard", "goal": "Show blind spot hazard"},
    {"scene_id": "S3_control", "goal": "Demonstrate safe spotter use"}
]

# Generate placeholder images
images = []
for s in scenes:
    filename = f"{s['scene_id']}.jpg"
    (IMG_DIR / filename).write_text("fake image bytes")
    images.append({
        "scene_id": s["scene_id"],
        "file": f"img/{filename}",
        "alt": f"{s['goal']} (training visual)",
        "caption": s["goal"]
    })

# Generate placeholder video
video_filename = "forklift_demo_8s.mp4"
(VIDEO_DIR / video_filename).write_text("fake video bytes")
video = {
    "scene_ids": [s["scene_id"] for s in scenes],
    "file": f"video/{video_filename}",
    "duration": 8,
    "caption": "Use a spotter when reversing."
}

# Write media.json next to this script (in media_agent/)
bundle = {"images": images, "video": video}
MEDIA_JSON_PATH.write_text(json.dumps(bundle, indent=2))

print(f"✅ Media bundle written to {MEDIA_JSON_PATH}")
