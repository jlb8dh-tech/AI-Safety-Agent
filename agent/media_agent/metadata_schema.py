"""
Structured Metadata Schema for Safety Content
Defines data models and schemas for safety training content metadata.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import hashlib


class ContentType(Enum):
    """Types of safety content."""
    IMAGE = "image"
    VIDEO = "video"
    DOCUMENT = "document"
    AUDIO = "audio"
    COMPOSITE = "composite"


class SafetyCategory(Enum):
    """Safety content categories."""
    HAZARD_IDENTIFICATION = "hazard_identification"
    CONTROL_MEASURES = "control_measures"
    PPE_USAGE = "ppe_usage"
    PROCEDURES = "procedures"
    EMERGENCY_RESPONSE = "emergency_response"
    TRAINING_DEMONSTRATION = "training_demonstration"
    COMPLIANCE = "compliance"
    AWARENESS = "awareness"


class HazardType(Enum):
    """Types of hazards."""
    BLIND_SPOT = "blind_spot"
    STRIKING = "striking"
    FALLING = "falling"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    FIRE = "fire"
    ERGONOMIC = "ergonomic"
    NOISE = "noise"
    MACHINE_OPERATION = "machine_operation"
    VEHICLE_OPERATION = "vehicle_operation"


class ContentStatus(Enum):
    """Processing status of content."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    TAGGED = "tagged"
    CLASSIFIED = "classified"
    READY = "ready"
    ERROR = "error"


@dataclass
class StandardsReference:
    """Safety standards and regulations reference."""
    standard_name: str  # e.g., "OSHA", "NIOSH", "ANSI"
    regulation_id: str  # e.g., "1910.178(n)(6)"
    description: Optional[str] = None
    url: Optional[str] = None


@dataclass
class ContentMetadata:
    """Comprehensive metadata for safety content."""
    # Basic identification
    content_id: str
    content_type: ContentType
    file_path: str
    file_name: str
    file_size: int
    file_hash: str
    
    # Temporal metadata
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    
    # Content description
    title: Optional[str] = None
    description: Optional[str] = None
    caption: Optional[str] = None
    alt_text: Optional[str] = None
    
    # Classification
    safety_category: Optional[SafetyCategory] = None
    hazard_types: List[HazardType] = field(default_factory=list)
    scene_id: Optional[str] = None
    scene_goal: Optional[str] = None
    
    # Standards and compliance
    standards: List[StandardsReference] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    
    # Automated tags
    auto_tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Media-specific metadata
    duration: Optional[float] = None  # for video/audio
    dimensions: Optional[Dict[str, int]] = None  # width, height for images/video
    frame_rate: Optional[float] = None  # for video
    codec: Optional[str] = None  # for video/audio
    
    # Relationships
    related_scenes: List[str] = field(default_factory=list)
    related_content_ids: List[str] = field(default_factory=list)
    
    # Processing metadata
    status: ContentStatus = ContentStatus.UPLOADED
    processing_notes: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Additional metadata
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization."""
        data = asdict(self)
        # Convert enums to their values
        data['content_type'] = self.content_type.value
        data['status'] = self.status.value
        if self.safety_category:
            data['safety_category'] = self.safety_category.value
        data['hazard_types'] = [ht.value for ht in self.hazard_types]
        data['standards'] = [asdict(std) for std in self.standards]
        
        # Convert datetime to ISO format strings
        data['uploaded_at'] = self.uploaded_at.isoformat()
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        if self.modified_at:
            data['modified_at'] = self.modified_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentMetadata':
        """Create from dictionary with proper deserialization."""
        # Convert string values back to enums
        data['content_type'] = ContentType(data['content_type'])
        data['status'] = ContentStatus(data['status'])
        if data.get('safety_category'):
            data['safety_category'] = SafetyCategory(data['safety_category'])
        if data.get('hazard_types'):
            data['hazard_types'] = [HazardType(ht) for ht in data['hazard_types']]
        if data.get('standards'):
            data['standards'] = [StandardsReference(**std) for std in data['standards']]
        
        # Convert ISO format strings back to datetime
        data['uploaded_at'] = datetime.fromisoformat(data['uploaded_at'])
        if data.get('processed_at'):
            data['processed_at'] = datetime.fromisoformat(data['processed_at'])
        if data.get('modified_at'):
            data['modified_at'] = datetime.fromisoformat(data['modified_at'])
        
        return cls(**data)
    
    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception:
            return ""


@dataclass
class MetadataCollection:
    """Collection of content metadata."""
    collection_id: str
    collection_name: str
    contents: List[ContentMetadata] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['contents'] = [content.to_dict() for content in self.contents]
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataCollection':
        """Create from dictionary."""
        data['contents'] = [ContentMetadata.from_dict(c) for c in data.get('contents', [])]
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)
    
    def save(self, file_path: Path):
        """Save collection to JSON file."""
        file_path.write_text(json.dumps(self.to_dict(), indent=2))
    
    @classmethod
    def load(cls, file_path: Path) -> 'MetadataCollection':
        """Load collection from JSON file."""
        return cls.from_dict(json.loads(file_path.read_text()))

