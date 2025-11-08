"""
Automated Tagging and Classification Engine for Safety Content
Analyzes content and automatically assigns tags, categories, and classifications.
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import json
from datetime import datetime

from metadata_schema import (
    ContentMetadata, SafetyCategory, HazardType, StandardsReference,
    ContentType, ContentStatus
)


class TaggingEngine:
    """Engine for automated tagging and classification of safety content."""
    
    def __init__(self, taxonomy_config_path: Optional[Path] = None):
        """Initialize tagging engine with taxonomy configuration."""
        if taxonomy_config_path is None:
            taxonomy_config_path = Path(__file__).parent / "taxonomy_config.json"
        
        with open(taxonomy_config_path) as f:
            self.taxonomy = json.load(f)
        
        # Build keyword indexes for faster lookup
        self._build_keyword_indexes()
    
    def _build_keyword_indexes(self):
        """Build reverse indexes for keyword matching."""
        self.category_keywords = {}
        for category, config in self.taxonomy["safety_categories"].items():
            self.category_keywords[category] = [
                kw.lower() for kw in config["keywords"]
            ]
        
        self.hazard_keywords = {}
        for hazard, config in self.taxonomy["hazard_types"].items():
            self.hazard_keywords[hazard] = [
                kw.lower() for kw in config["keywords"]
            ]
        
        self.standards_patterns = {}
        for std_name, config in self.taxonomy["standards_mapping"].items():
            self.standards_patterns[std_name] = [
                pattern.lower() for pattern in config["patterns"]
            ]
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        if not text:
            return []
        
        text_lower = text.lower()
        keywords = set()
        
        # Check all common tags
        for tag_category, tag_list in self.taxonomy["common_tags"].items():
            for tag in tag_list:
                if tag.lower() in text_lower:
                    keywords.add(tag)
        
        # Extract safety-specific terms
        all_keywords = []
        for category_config in self.category_keywords.values():
            all_keywords.extend(category_config)
        for hazard_config in self.hazard_keywords.values():
            all_keywords.extend(hazard_config)
        
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.add(keyword)
        
        return sorted(list(keywords))
    
    def classify_safety_category(self, text: str, scene_goal: Optional[str] = None) -> Tuple[Optional[SafetyCategory], float]:
        """Classify content into safety category with confidence score."""
        if not text and not scene_goal:
            return None, 0.0
        
        combined_text = f"{text or ''} {scene_goal or ''}".lower()
        scores = {}
        
        for category_name, keywords in self.category_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword in combined_text:
                    matches += 1
                    # Longer keywords get higher weight
                    score += len(keyword) * 0.1
            
            # Normalize by number of keywords
            if len(keywords) > 0:
                score = score / len(keywords)
                scores[category_name] = score
        
        if not scores:
            return None, 0.0
        
        # Get category with highest score
        best_category = max(scores.items(), key=lambda x: x[1])
        if best_category[1] > 0.1:  # Minimum confidence threshold
            return SafetyCategory(best_category[0]), min(best_category[1], 1.0)
        
        return None, 0.0
    
    def detect_hazard_types(self, text: str, scene_goal: Optional[str] = None) -> List[Tuple[HazardType, float]]:
        """Detect hazard types with confidence scores."""
        if not text and not scene_goal:
            return []
        
        combined_text = f"{text or ''} {scene_goal or ''}".lower()
        hazard_scores = []
        
        for hazard_name, keywords in self.hazard_keywords.items():
            score = 0.0
            matches = 0
            for keyword in keywords:
                if keyword in combined_text:
                    matches += 1
                    score += len(keyword) * 0.1
            
            if len(keywords) > 0:
                score = score / len(keywords)
            
            if score > 0.1:  # Minimum threshold
                hazard_scores.append((HazardType(hazard_name), min(score, 1.0)))
        
        # Sort by confidence and return top matches
        hazard_scores.sort(key=lambda x: x[1], reverse=True)
        return hazard_scores[:5]  # Return top 5
    
    def extract_standards_references(self, text: str) -> List[StandardsReference]:
        """Extract safety standards references from text."""
        if not text:
            return []
        
        text_lower = text.lower()
        standards = []
        
        # Pattern for OSHA regulations (e.g., "1910.178(n)(6)")
        osha_pattern = r'(?:osha|29\s*cfr)?\s*(?:1910|1926)\.?\s*\d+(?:\.\d+)*\s*(?:\([a-z0-9]+\))*'
        
        # Find OSHA references
        osha_matches = re.findall(osha_pattern, text_lower, re.IGNORECASE)
        for match in osha_matches:
            # Clean up the match
            regulation = re.sub(r'\s+', '', match)
            regulation = re.sub(r'osha|29cfr', '', regulation, flags=re.IGNORECASE).strip()
            if regulation:
                standards.append(StandardsReference(
                    standard_name="OSHA",
                    regulation_id=regulation,
                    description=f"OSHA Regulation {regulation}"
                ))
        
        # Check for other standards mentions
        for std_name, patterns in self.standards_patterns.items():
            for pattern in patterns:
                if pattern in text_lower and std_name not in [s.standard_name for s in standards]:
                    # Try to extract regulation ID
                    # Look for numbers or codes after the standard name
                    std_match = re.search(
                        rf'{re.escape(pattern)}\s*([A-Z0-9.\-]+)',
                        text,
                        re.IGNORECASE
                    )
                    regulation_id = std_match.group(1) if std_match else "general"
                    
                    standards.append(StandardsReference(
                        standard_name=std_name,
                        regulation_id=regulation_id,
                        description=f"{std_name} standard"
                    ))
        
        return standards
    
    def generate_auto_tags(self, metadata: ContentMetadata) -> List[str]:
        """Generate automatic tags based on metadata."""
        tags = set()
        
        # Add content type tag
        tags.add(f"type:{metadata.content_type.value}")
        
        # Add safety category tag
        if metadata.safety_category:
            tags.add(f"category:{metadata.safety_category.value}")
        
        # Add hazard type tags
        for hazard in metadata.hazard_types:
            tags.add(f"hazard:{hazard.value}")
        
        # Add standards tags
        for std in metadata.standards:
            tags.add(f"standard:{std.standard_name.lower()}")
            tags.add(f"regulation:{std.regulation_id}")
        
        # Extract keywords from text fields
        text_fields = [
            metadata.title,
            metadata.description,
            metadata.caption,
            metadata.alt_text,
            metadata.scene_goal
        ]
        combined_text = " ".join([tf for tf in text_fields if tf])
        
        keywords = self.extract_keywords_from_text(combined_text)
        tags.update(keywords)
        
        # Add scene-related tags
        if metadata.scene_id:
            tags.add(f"scene:{metadata.scene_id}")
        
        # Add compliance tags if standards are present
        if metadata.standards:
            tags.add("compliance")
        
        return sorted(list(tags))
    
    def classify_content(self, metadata: ContentMetadata) -> ContentMetadata:
        """Perform full classification and tagging of content."""
        # Combine all text fields for analysis
        text_fields = [
            metadata.title,
            metadata.description,
            metadata.caption,
            metadata.alt_text,
            metadata.scene_goal
        ]
        combined_text = " ".join([tf for tf in text_fields if tf])
        
        # Classify safety category
        category, category_confidence = self.classify_safety_category(
            combined_text, metadata.scene_goal
        )
        if category:
            metadata.safety_category = category
            metadata.confidence_scores["safety_category"] = category_confidence
        
        # Detect hazard types
        hazard_detections = self.detect_hazard_types(combined_text, metadata.scene_goal)
        metadata.hazard_types = [h for h, _ in hazard_detections]
        for hazard, confidence in hazard_detections:
            metadata.confidence_scores[f"hazard:{hazard.value}"] = confidence
        
        # Extract standards references
        standards = self.extract_standards_references(combined_text)
        metadata.standards.extend(standards)
        
        # Generate auto tags
        metadata.auto_tags = self.generate_auto_tags(metadata)
        
        # Extract keywords
        metadata.keywords = self.extract_keywords_from_text(combined_text)
        
        # Update status
        metadata.status = ContentStatus.CLASSIFIED
        metadata.processed_at = datetime.now()
        
        return metadata

