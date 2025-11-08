"""
Media Generation Script
Generates placeholder media files and uses the ingestion pipeline to create structured metadata.
"""

import json
from pathlib import Path
from ingestion_pipeline import IngestionPipeline
from metadata_schema import MetadataCollection

# Always work relative to THIS file's folder (media_agent/)
BASE_DIR = Path(__file__).parent.resolve()

ASSETS_DIR = BASE_DIR / "assets"
IMG_DIR = ASSETS_DIR / "img"
VIDEO_DIR = ASSETS_DIR / "video"
MEDIA_JSON_PATH = BASE_DIR / "media.json"
SCENE_PLAN_PATH = BASE_DIR / "scene_plan.json"

IMG_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Load scene plan if available, otherwise use default scenes
scenes = []
if SCENE_PLAN_PATH.exists():
    try:
        with open(SCENE_PLAN_PATH) as f:
            scene_plan = json.load(f)
            scenes = scene_plan.get("scenes", [])
    except Exception:
        scenes = []

# Fallback to dummy scenes if no scene plan
if not scenes:
    scenes = [
        {"scene_id": "S1_intro", "goal": "Introduce warehouse environment"},
        {"scene_id": "S2_hazard", "goal": "Show blind spot hazard"},
        {"scene_id": "S3_control", "goal": "Demonstrate safe spotter use"}
    ]

# Generate placeholder images
for s in scenes:
    filename = f"{s['scene_id']}.jpg"
    img_path = IMG_DIR / filename
    if not img_path.exists():
        img_path.write_text("fake image bytes")

# Generate placeholder video
video_filename = "forklift_demo_8s.mp4"
video_path = VIDEO_DIR / video_filename
if not video_path.exists():
    video_path.write_text("fake video bytes")

# Initialize ingestion pipeline
print("🔄 Processing media through ingestion pipeline...")
pipeline = IngestionPipeline(
    assets_dir=ASSETS_DIR,
    metadata_dir=BASE_DIR / "metadata"
)

# Ingest all generated media files
image_metadata = []
for s in scenes:
    filename = f"{s['scene_id']}.jpg"
    img_path = IMG_DIR / filename
    if img_path.exists():
        try:
            metadata = pipeline.ingest_file(
                img_path,
                caption=s.get("goal", ""),
                scene_plan_path=SCENE_PLAN_PATH
            )
            image_metadata.append(metadata)
        except Exception as e:
            print(f"Warning: Error processing {img_path}: {e}")

# Ingest video
video_metadata = None
if video_path.exists():
    try:
        video_metadata = pipeline.ingest_file(
            video_path,
            caption="Use a spotter when reversing.",
            scene_plan_path=SCENE_PLAN_PATH
        )
        # Set related scenes for video
        if video_metadata:
            video_metadata.related_scenes = [s["scene_id"] for s in scenes]
            video_metadata.duration = 8.0
    except Exception as e:
        print(f"Warning: Error processing {video_path}: {e}")

# Save metadata collection
collection = MetadataCollection(
    collection_id="generated_media",
    collection_name="Generated Media"
)
collection.contents.extend(image_metadata)
if video_metadata:
    collection.contents.append(video_metadata)

metadata_dir = BASE_DIR / "metadata"
metadata_dir.mkdir(parents=True, exist_ok=True)
collection_path = metadata_dir / "generated_media.json"
collection.save(collection_path)

# Generate legacy media.json format for backward compatibility
legacy_media = {
    "images": [],
    "video": None
}

for img_meta in image_metadata:
    legacy_media["images"].append({
        "scene_id": img_meta.scene_id,
        "file": img_meta.file_path.replace("assets/", ""),
        "alt": img_meta.alt_text or img_meta.caption or "",
        "caption": img_meta.caption or img_meta.title or ""
    })

if video_metadata:
    legacy_media["video"] = {
        "scene_ids": video_metadata.related_scenes or [s["scene_id"] for s in scenes],
        "file": video_metadata.file_path.replace("assets/", ""),
        "duration": video_metadata.duration or 8,
        "caption": video_metadata.caption or ""
    }

MEDIA_JSON_PATH.write_text(json.dumps(legacy_media, indent=2))

print(f"✅ Media bundle written to {MEDIA_JSON_PATH}")
print(f"✅ Structured metadata saved to {collection_path}")
print(f"   Processed {len(image_metadata)} images and {1 if video_metadata else 0} video(s)")
