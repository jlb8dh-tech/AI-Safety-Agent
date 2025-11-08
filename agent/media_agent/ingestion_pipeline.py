"""
Ingestion Pipeline for Safety Content
Processes uploaded content and extracts structured metadata.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import hashlib
import mimetypes

from metadata_schema import (
    ContentMetadata, ContentType, ContentStatus, MetadataCollection
)
from tagging_engine import TaggingEngine


class IngestionPipeline:
    """Pipeline for ingesting and processing safety content."""
    
    def __init__(
        self,
        assets_dir: Path,
        metadata_dir: Optional[Path] = None,
        taxonomy_config_path: Optional[Path] = None
    ):
        """Initialize ingestion pipeline."""
        self.assets_dir = Path(assets_dir)
        self.metadata_dir = metadata_dir or self.assets_dir.parent / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.tagging_engine = TaggingEngine(taxonomy_config_path)
        
        # Supported file extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        self.document_extensions = {'.pdf', '.doc', '.docx', '.txt', '.md'}
        self.audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a'}
    
    def detect_content_type(self, file_path: Path) -> ContentType:
        """Detect content type from file extension."""
        ext = file_path.suffix.lower()
        
        if ext in self.image_extensions:
            return ContentType.IMAGE
        elif ext in self.video_extensions:
            return ContentType.VIDEO
        elif ext in self.document_extensions:
            return ContentType.DOCUMENT
        elif ext in self.audio_extensions:
            return ContentType.AUDIO
        else:
            return ContentType.COMPOSITE
    
    def extract_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic file metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        file_size = stat.st_size
        
        # Compute file hash
        file_hash = self._compute_file_hash(file_path)
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            "file_path": str(file_path.relative_to(self.assets_dir.parent)),
            "file_name": file_path.name,
            "file_size": file_size,
            "file_hash": file_hash,
            "mime_type": mime_type
        }
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            return ""
    
    def extract_media_metadata(self, file_path: Path, content_type: ContentType) -> Dict[str, Any]:
        """Extract media-specific metadata."""
        metadata = {}
        
        if content_type == ContentType.IMAGE:
            # For images, could use PIL to get dimensions
            # For now, return placeholder
            metadata["dimensions"] = {"width": 0, "height": 0}
        
        elif content_type == ContentType.VIDEO:
            # For videos, could use ffmpeg or similar to get duration, dimensions, codec
            # For now, return placeholders
            metadata["duration"] = 0.0
            metadata["dimensions"] = {"width": 0, "height": 0}
            metadata["frame_rate"] = 30.0
            metadata["codec"] = "unknown"
        
        elif content_type == ContentType.AUDIO:
            # For audio, could extract duration, codec, bitrate
            metadata["duration"] = 0.0
            metadata["codec"] = "unknown"
        
        return metadata
    
    def generate_content_id(self, file_path: Path, file_hash: str) -> str:
        """Generate unique content ID."""
        # Use file hash as base for content ID
        return f"content_{file_hash[:16]}"
    
    def load_scene_context(self, scene_plan_path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
        """Load scene context from scene plan."""
        if scene_plan_path is None:
            scene_plan_path = Path(__file__).parent / "scene_plan.json"
        
        if not scene_plan_path.exists():
            return {}
        
        try:
            with open(scene_plan_path) as f:
                scene_plan = json.load(f)
            
            # Build scene lookup dictionary
            scenes = {}
            for scene in scene_plan.get("scenes", []):
                scene_id = scene.get("scene_id")
                if scene_id:
                    scenes[scene_id] = scene
            
            return scenes
        except Exception:
            return {}
    
    def match_scene_context(self, file_path: Path, scenes: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Match file to scene context based on filename or metadata."""
        filename = file_path.stem  # filename without extension
        
        # Try to match scene_id from filename (e.g., "S1_intro.jpg" -> "S1_intro")
        if filename in scenes:
            return scenes[filename]
        
        # Try partial matches
        for scene_id, scene_data in scenes.items():
            if scene_id in filename or filename in scene_id:
                return scene_data
        
        return None
    
    def ingest_file(
        self,
        file_path: Path,
        title: Optional[str] = None,
        description: Optional[str] = None,
        caption: Optional[str] = None,
        scene_plan_path: Optional[Path] = None
    ) -> ContentMetadata:
        """Ingest a single file and extract metadata."""
        file_path = Path(file_path)
        
        # Update status
        status = ContentStatus.PROCESSING
        
        # Extract basic metadata
        basic_meta = self.extract_basic_metadata(file_path)
        content_type = self.detect_content_type(file_path)
        
        # Extract media-specific metadata
        media_meta = self.extract_media_metadata(file_path, content_type)
        
        # Generate content ID
        content_id = self.generate_content_id(file_path, basic_meta["file_hash"])
        
        # Load and match scene context
        scenes = self.load_scene_context(scene_plan_path)
        scene_context = self.match_scene_context(file_path, scenes)
        
        # Create metadata object
        metadata = ContentMetadata(
            content_id=content_id,
            content_type=content_type,
            file_path=basic_meta["file_path"],
            file_name=basic_meta["file_name"],
            file_size=basic_meta["file_size"],
            file_hash=basic_meta["file_hash"],
            uploaded_at=datetime.now(),
            title=title,
            description=description,
            caption=caption,
            status=status,
            **media_meta
        )
        
        # Add scene context if matched
        if scene_context:
            metadata.scene_id = scene_context.get("scene_id")
            metadata.scene_goal = scene_context.get("goal")
            
            # Add standards from scene context
            standards_tags = scene_context.get("standards_tags", [])
            for std_tag in standards_tags:
                # Parse standards tags (e.g., "OSHA 1910.178(n)(6)")
                std_ref = self.tagging_engine.extract_standards_references(std_tag)
                metadata.standards.extend(std_ref)
        
        # Perform automated classification and tagging
        metadata = self.tagging_engine.classify_content(metadata)
        
        return metadata
    
    def ingest_directory(
        self,
        directory: Path,
        scene_plan_path: Optional[Path] = None,
        recursive: bool = True
    ) -> List[ContentMetadata]:
        """Ingest all files in a directory."""
        directory = Path(directory)
        metadata_list = []
        
        # Find all files
        pattern = "**/*" if recursive else "*"
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    metadata = self.ingest_file(file_path, scene_plan_path=scene_plan_path)
                    metadata_list.append(metadata)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
        
        return metadata_list
    
    def save_metadata(self, metadata: ContentMetadata, collection_name: str = "default"):
        """Save metadata to collection."""
        collection_path = self.metadata_dir / f"{collection_name}.json"
        
        # Load existing collection or create new one
        if collection_path.exists():
            collection = MetadataCollection.load(collection_path)
        else:
            collection = MetadataCollection(
                collection_id=collection_name,
                collection_name=collection_name
            )
        
        # Update or add metadata
        existing_idx = None
        for i, content in enumerate(collection.contents):
            if content.content_id == metadata.content_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            collection.contents[existing_idx] = metadata
        else:
            collection.contents.append(metadata)
        
        collection.updated_at = datetime.now()
        collection.save(collection_path)
        
        return collection
    
    def load_metadata_collection(self, collection_name: str = "default") -> MetadataCollection:
        """Load metadata collection."""
        collection_path = self.metadata_dir / f"{collection_name}.json"
        
        if collection_path.exists():
            return MetadataCollection.load(collection_path)
        else:
            return MetadataCollection(
                collection_id=collection_name,
                collection_name=collection_name
            )

