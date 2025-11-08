"""
Main script for ingesting safety content using the structured metadata pipeline.
"""

import json
from pathlib import Path
from ingestion_pipeline import IngestionPipeline
from metadata_schema import MetadataCollection


def main():
    """Main ingestion script."""
    # Setup paths
    BASE_DIR = Path(__file__).parent.resolve()
    ASSETS_DIR = BASE_DIR / "assets"
    IMG_DIR = ASSETS_DIR / "img"
    VIDEO_DIR = ASSETS_DIR / "video"
    SCENE_PLAN_PATH = BASE_DIR / "scene_plan.json"
    
    # Initialize ingestion pipeline
    pipeline = IngestionPipeline(
        assets_dir=ASSETS_DIR,
        metadata_dir=BASE_DIR / "metadata"
    )
    
    print("🔄 Starting safety content ingestion pipeline...")
    print(f"   Assets directory: {ASSETS_DIR}")
    print(f"   Scene plan: {SCENE_PLAN_PATH}")
    print()
    
    # Ingest images
    print("📸 Ingesting images...")
    image_metadata = []
    if IMG_DIR.exists():
        for img_file in IMG_DIR.glob("*.jpg"):
            try:
                metadata = pipeline.ingest_file(
                    img_file,
                    scene_plan_path=SCENE_PLAN_PATH
                )
                image_metadata.append(metadata)
                print(f"   ✅ {img_file.name}")
                print(f"      ID: {metadata.content_id}")
                print(f"      Category: {metadata.safety_category.value if metadata.safety_category else 'None'}")
                print(f"      Tags: {', '.join(metadata.auto_tags[:5])}")
            except Exception as e:
                print(f"   ❌ Error ingesting {img_file.name}: {e}")
    
    # Ingest videos
    print("\n🎥 Ingesting videos...")
    video_metadata = []
    if VIDEO_DIR.exists():
        for vid_file in VIDEO_DIR.glob("*.mp4"):
            try:
                metadata = pipeline.ingest_file(
                    vid_file,
                    scene_plan_path=SCENE_PLAN_PATH
                )
                video_metadata.append(metadata)
                print(f"   ✅ {vid_file.name}")
                print(f"      ID: {metadata.content_id}")
                print(f"      Category: {metadata.safety_category.value if metadata.safety_category else 'None'}")
                print(f"      Tags: {', '.join(metadata.auto_tags[:5])}")
            except Exception as e:
                print(f"   ❌ Error ingesting {vid_file.name}: {e}")
    
    # Save to metadata collection
    print("\n💾 Saving metadata collection...")
    collection = MetadataCollection(
        collection_id="safety_training_content",
        collection_name="Safety Training Content"
    )
    
    for metadata in image_metadata + video_metadata:
        collection.contents.append(metadata)
    
    collection_path = BASE_DIR / "metadata" / "safety_training_content.json"
    collection.save(collection_path)
    print(f"   ✅ Saved to {collection_path}")
    
    # Generate legacy media.json format for backward compatibility
    print("\n📋 Generating legacy media.json format...")
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
    
    # Find video if exists
    if video_metadata:
        vid_meta = video_metadata[0]
        legacy_media["video"] = {
            "scene_ids": vid_meta.related_scenes or [vid_meta.scene_id] if vid_meta.scene_id else [],
            "file": vid_meta.file_path.replace("assets/", ""),
            "duration": vid_meta.duration or 0,
            "caption": vid_meta.caption or vid_meta.title or ""
        }
    
    legacy_path = BASE_DIR / "media.json"
    legacy_path.write_text(json.dumps(legacy_media, indent=2))
    print(f"   ✅ Saved to {legacy_path}")
    
    # Print summary
    print("\n📊 Ingestion Summary:")
    print(f"   Images processed: {len(image_metadata)}")
    print(f"   Videos processed: {len(video_metadata)}")
    print(f"   Total content: {len(collection.contents)}")
    print("\n✅ Ingestion pipeline completed successfully!")


if __name__ == "__main__":
    main()

