"""
Utility script for querying and searching metadata collections.
"""

import json
from pathlib import Path
from typing import List, Optional
from metadata_schema import MetadataCollection, SafetyCategory, HazardType, ContentType


def load_collection(collection_path: Path) -> MetadataCollection:
    """Load metadata collection from file."""
    return MetadataCollection.load(collection_path)


def search_by_category(
    collection: MetadataCollection,
    category: SafetyCategory
) -> List:
    """Search content by safety category."""
    return [
        content for content in collection.contents
        if content.safety_category == category
    ]


def search_by_hazard(
    collection: MetadataCollection,
    hazard_type: HazardType
) -> List:
    """Search content by hazard type."""
    return [
        content for content in collection.contents
        if hazard_type in content.hazard_types
    ]


def search_by_standard(
    collection: MetadataCollection,
    standard_name: str
) -> List:
    """Search content by safety standard."""
    return [
        content for content in collection.contents
        if any(s.standard_name == standard_name for s in content.standards)
    ]


def search_by_tag(
    collection: MetadataCollection,
    tag: str
) -> List:
    """Search content by tag."""
    return [
        content for content in collection.contents
        if tag in content.auto_tags or tag in content.keywords
    ]


def search_by_scene(
    collection: MetadataCollection,
    scene_id: str
) -> List:
    """Search content by scene ID."""
    return [
        content for content in collection.contents
        if content.scene_id == scene_id or scene_id in content.related_scenes
    ]


def search_by_keyword(
    collection: MetadataCollection,
    keyword: str
) -> List:
    """Search content by keyword in any text field."""
    keyword_lower = keyword.lower()
    results = []
    
    for content in collection.contents:
        text_fields = [
            content.title,
            content.description,
            content.caption,
            content.alt_text,
            content.scene_goal
        ]
        combined_text = " ".join([tf for tf in text_fields if tf]).lower()
        
        if keyword_lower in combined_text:
            results.append(content)
    
    return results


def print_content_summary(content, detailed: bool = False):
    """Print summary of content metadata."""
    print(f"  📄 {content.file_name}")
    print(f"     ID: {content.content_id}")
    print(f"     Type: {content.content_type.value}")
    if content.safety_category:
        print(f"     Category: {content.safety_category.value}")
    if content.hazard_types:
        print(f"     Hazards: {', '.join([ht.value for ht in content.hazard_types])}")
    if content.standards:
        stds = [f"{s.standard_name} {s.regulation_id}" for s in content.standards]
        print(f"     Standards: {', '.join(stds)}")
    if content.auto_tags:
        print(f"     Tags: {', '.join(content.auto_tags[:5])}")
    
    if detailed:
        if content.title:
            print(f"     Title: {content.title}")
        if content.description:
            print(f"     Description: {content.description}")
        if content.caption:
            print(f"     Caption: {content.caption}")
        if content.confidence_scores:
            print(f"     Confidence: {content.confidence_scores}")


def main():
    """Main query interface."""
    import sys
    
    BASE_DIR = Path(__file__).parent.resolve()
    metadata_dir = BASE_DIR / "metadata"
    
    # Find available collections
    collections = list(metadata_dir.glob("*.json"))
    
    if not collections:
        print("❌ No metadata collections found.")
        print(f"   Expected in: {metadata_dir}")
        return
    
    # Use first collection or specified one
    if len(sys.argv) > 1:
        collection_path = Path(sys.argv[1])
    else:
        collection_path = collections[0]
    
    if not collection_path.exists():
        print(f"❌ Collection not found: {collection_path}")
        return
    
    print(f"📚 Loading collection: {collection_path.name}")
    collection = load_collection(collection_path)
    print(f"   Found {len(collection.contents)} content items\n")
    
    # Interactive query interface
    while True:
        print("\n🔍 Query Options:")
        print("  1. Search by category")
        print("  2. Search by hazard type")
        print("  3. Search by standard")
        print("  4. Search by tag")
        print("  5. Search by scene")
        print("  6. Search by keyword")
        print("  7. Show all content")
        print("  8. Exit")
        
        choice = input("\nEnter choice (1-8): ").strip()
        
        if choice == "1":
            print("\nAvailable categories:")
            for i, cat in enumerate(SafetyCategory, 1):
                print(f"  {i}. {cat.value}")
            cat_choice = input("Enter category number: ").strip()
            try:
                category = list(SafetyCategory)[int(cat_choice) - 1]
                results = search_by_category(collection, category)
                print(f"\n📊 Found {len(results)} results:")
                for content in results:
                    print_content_summary(content)
            except (ValueError, IndexError):
                print("❌ Invalid category")
        
        elif choice == "2":
            print("\nAvailable hazard types:")
            for i, hazard in enumerate(HazardType, 1):
                print(f"  {i}. {hazard.value}")
            hazard_choice = input("Enter hazard number: ").strip()
            try:
                hazard = list(HazardType)[int(hazard_choice) - 1]
                results = search_by_hazard(collection, hazard)
                print(f"\n📊 Found {len(results)} results:")
                for content in results:
                    print_content_summary(content)
            except (ValueError, IndexError):
                print("❌ Invalid hazard type")
        
        elif choice == "3":
            standard = input("Enter standard name (e.g., OSHA, NIOSH): ").strip()
            results = search_by_standard(collection, standard)
            print(f"\n📊 Found {len(results)} results:")
            for content in results:
                print_content_summary(content)
        
        elif choice == "4":
            tag = input("Enter tag to search for: ").strip()
            results = search_by_tag(collection, tag)
            print(f"\n📊 Found {len(results)} results:")
            for content in results:
                print_content_summary(content)
        
        elif choice == "5":
            scene_id = input("Enter scene ID (e.g., S1_intro): ").strip()
            results = search_by_scene(collection, scene_id)
            print(f"\n📊 Found {len(results)} results:")
            for content in results:
                print_content_summary(content)
        
        elif choice == "6":
            keyword = input("Enter keyword to search for: ").strip()
            results = search_by_keyword(collection, keyword)
            print(f"\n📊 Found {len(results)} results:")
            for content in results:
                print_content_summary(content)
        
        elif choice == "7":
            print(f"\n📊 All {len(collection.contents)} content items:")
            for content in collection.contents:
                print_content_summary(content)
                print()
        
        elif choice == "8":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()

