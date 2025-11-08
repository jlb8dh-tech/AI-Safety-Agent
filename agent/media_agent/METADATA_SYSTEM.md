# Structured Metadata System for Safety Content

## Overview

This system transforms the ingestion pipeline into a comprehensive data-to-metadata system with automated tagging and classification capabilities for safety training content.

## Architecture

### Components

1. **Metadata Schema** (`metadata_schema.py`)
   - Defines structured data models for safety content
   - Supports images, videos, documents, and audio
   - Includes comprehensive metadata fields (categories, hazards, standards, tags)

2. **Ingestion Pipeline** (`ingestion_pipeline.py`)
   - Processes uploaded content files
   - Extracts basic and media-specific metadata
   - Matches content to scene context
   - Integrates with tagging engine

3. **Tagging Engine** (`tagging_engine.py`)
   - Automated classification of safety categories
   - Hazard type detection
   - Standards reference extraction
   - Keyword extraction and auto-tagging

4. **Taxonomy Configuration** (`taxonomy_config.json`)
   - Safety categories and keywords
   - Hazard types and detection patterns
   - Standards mapping (OSHA, NIOSH, ANSI, ISO)
   - Common tags and classifications

## Usage

### Basic Ingestion

```python
from pathlib import Path
from ingestion_pipeline import IngestionPipeline

# Initialize pipeline
pipeline = IngestionPipeline(
    assets_dir=Path("assets"),
    metadata_dir=Path("metadata")
)

# Ingest a single file
metadata = pipeline.ingest_file(
    file_path=Path("assets/img/S1_intro.jpg"),
    title="Warehouse Introduction",
    description="Introduction to warehouse safety",
    caption="Warehouse environment overview"
)

# Save to collection
pipeline.save_metadata(metadata, collection_name="safety_training")
```

### Batch Ingestion

```python
# Ingest all files in a directory
metadata_list = pipeline.ingest_directory(
    directory=Path("assets"),
    scene_plan_path=Path("scene_plan.json"),
    recursive=True
)

# Save all to collection
for metadata in metadata_list:
    pipeline.save_metadata(metadata, collection_name="safety_training")
```

### Using the Command Line Script

```bash
# Run the ingestion script
python ingest_media.py

# This will:
# 1. Process all files in assets/ directory
# 2. Extract metadata and apply classifications
# 3. Generate structured metadata collections
# 4. Create backward-compatible media.json
```

## Metadata Structure

### ContentMetadata Fields

- **Basic Information**: content_id, content_type, file_path, file_name, file_size, file_hash
- **Temporal**: uploaded_at, processed_at, modified_at
- **Description**: title, description, caption, alt_text
- **Classification**: safety_category, hazard_types, scene_id, scene_goal
- **Standards**: standards (list of StandardsReference), compliance_tags
- **Tags**: auto_tags, keywords
- **Media**: duration, dimensions, frame_rate, codec
- **Relationships**: related_scenes, related_content_ids
- **Processing**: status, processing_notes, confidence_scores

### Safety Categories

- `hazard_identification`: Content showing hazards
- `control_measures`: Safety controls and protections
- `ppe_usage`: Personal protective equipment
- `procedures`: Safety procedures and protocols
- `emergency_response`: Emergency procedures
- `training_demonstration`: Training demonstrations
- `compliance`: Regulatory compliance
- `awareness`: Safety awareness content

### Hazard Types

- `blind_spot`: Blind spot hazards
- `striking`: Striking and impact hazards
- `falling`: Fall hazards
- `electrical`: Electrical hazards
- `chemical`: Chemical hazards
- `fire`: Fire hazards
- `ergonomic`: Ergonomic hazards
- `noise`: Noise hazards
- `machine_operation`: Machine operation hazards
- `vehicle_operation`: Vehicle operation hazards

## Automated Tagging

The tagging engine automatically:

1. **Classifies** content into safety categories based on keywords and context
2. **Detects** hazard types mentioned in descriptions
3. **Extracts** standards references (e.g., "OSHA 1910.178(n)(6)")
4. **Generates** tags from content type, category, hazards, and keywords
5. **Assigns** confidence scores for classifications

## Standards References

The system automatically extracts references to:

- **OSHA**: Occupational Safety and Health Administration regulations
- **NIOSH**: National Institute for Occupational Safety and Health
- **ANSI**: American National Standards Institute
- **ISO**: International Organization for Standardization

## Querying Metadata

```python
from metadata_schema import MetadataCollection
from pathlib import Path

# Load collection
collection = MetadataCollection.load(Path("metadata/safety_training_content.json"))

# Query by category
hazard_content = [
    c for c in collection.contents
    if c.safety_category == SafetyCategory.HAZARD_IDENTIFICATION
]

# Query by hazard type
blind_spot_content = [
    c for c in collection.contents
    if HazardType.BLIND_SPOT in c.hazard_types
]

# Query by standards
osha_content = [
    c for c in collection.contents
    if any(s.standard_name == "OSHA" for s in c.standards)
]

# Query by tags
tagged_content = [
    c for c in collection.contents
    if "forklift" in c.auto_tags
]
```

## Configuration

### Customizing Taxonomies

Edit `taxonomy_config.json` to:

- Add new safety categories
- Define new hazard types
- Update keyword mappings
- Add standards patterns
- Customize common tags

### Extending the System

1. **Add new content types**: Extend `ContentType` enum in `metadata_schema.py`
2. **Add classification rules**: Update `tagging_engine.py` methods
3. **Custom metadata extraction**: Extend `extract_media_metadata()` in `ingestion_pipeline.py`
4. **New standards**: Add patterns to `taxonomy_config.json`

## Integration

The system maintains backward compatibility with the existing `media.json` format while providing rich structured metadata. The `generate_media.py` script has been updated to use the new pipeline automatically.

## Output

### Structured Metadata Collection

Metadata is saved as JSON collections in the `metadata/` directory:

```json
{
  "collection_id": "safety_training_content",
  "collection_name": "Safety Training Content",
  "contents": [
    {
      "content_id": "content_abc123...",
      "content_type": "image",
      "safety_category": "hazard_identification",
      "hazard_types": ["blind_spot"],
      "auto_tags": ["type:image", "category:hazard_identification", "hazard:blind_spot", "forklift"],
      "standards": [
        {
          "standard_name": "OSHA",
          "regulation_id": "1910.178(n)(6)",
          "description": "OSHA Regulation 1910.178(n)(6)"
        }
      ]
    }
  ]
}
```

### Legacy Format

The system also generates the legacy `media.json` format for backward compatibility with existing tools.

## Future Enhancements

- Image analysis using computer vision for automatic object detection
- Video analysis for scene segmentation and action recognition
- NLP-based content analysis for improved classification
- Integration with external safety standards databases
- Advanced search and filtering capabilities
- Content similarity detection
- Automated content recommendations

