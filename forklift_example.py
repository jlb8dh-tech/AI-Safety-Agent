#!/usr/bin/env python3
"""
Forklift Operation Training Example

This script demonstrates the enhanced Black Unicorn Agent capabilities for
workplace safety training, specifically forklift operation training.

Example Usage:
Input: "Forklift operation hazards"
Output: "Intro → Common Risks → Case Study → Safety Procedures → Demonstration → Quiz"

Standards Lookup:
Input: "Forklift operation"
Output: "1910.178 – Powered Industrial Trucks"
"""

import asyncio
import json
from pathlib import Path
from agent.pipelines.outline import OutlinePipeline
from agent.pipelines.standards import StandardsLookupPipeline, StandardsOrganization
from agent.pipelines.data_flows import DataFlowManager, TransformNode, FilterNode, RouterNode
from agent.utils.helpers import setup_logging


async def demonstrate_forklift_training():
    """Demonstrate forklift operation training workflow"""
    
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("🚛 Starting Forklift Operation Training Demonstration")
    
    print("=" * 80)
    print("🚛 FORKLIFT OPERATION TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Create training outline from user prompt
    print("\n📚 STEP 1: Creating Training Outline from User Prompt")
    print("-" * 60)
    
    user_prompt = "Forklift operation hazards"
    print(f"User Input: '{user_prompt}'")
    
    # Create outline pipeline
    outline_pipeline = OutlinePipeline()
    
    # Generate safety training outline
    training_outline = outline_pipeline.create_safety_training_outline(user_prompt)
    
    print(f"\nGenerated Training Flow: {training_outline['flow_summary']}")
    print(f"Total Duration: {training_outline['total_duration']} minutes")
    print(f"Topic Category: {training_outline['topic_category']}")
    
    print("\nDetailed Topics:")
    for i, topic in enumerate(training_outline['topics'], 1):
        print(f"  {i}. {topic['title']} ({topic['type']}) - {topic['duration']} min")
        print(f"     {topic['description']}")
    
    # Step 2: Look up relevant safety standards
    print("\n📋 STEP 2: Looking Up Relevant Safety Standards")
    print("-" * 60)
    
    standards_topic = "Forklift operation"
    print(f"Standards Lookup Input: '{standards_topic}'")
    
    # Create standards pipeline
    async with StandardsLookupPipeline() as standards_pipeline:
        # Look up workplace safety standards
        relevant_standards = standards_pipeline.lookup_workplace_safety_standards(standards_topic)
        
        print(f"\nFound {len(relevant_standards)} relevant standards:")
        
        for i, standard in enumerate(relevant_standards, 1):
            print(f"\n  {i}. {standard.standard_number} – {standard.title}")
            print(f"     Organization: {standard.organization.value.upper()}")
            print(f"     Compliance Level: {standard.compliance_level.value.title()}")
            print(f"     Description: {standard.description}")
            
            if standard.requirements:
                print(f"     Key Requirements:")
                for req in standard.requirements[:3]:  # Show first 3 requirements
                    print(f"       • {req}")
            
            if standard.penalties:
                print(f"     Penalties: {standard.penalties[0]}")
    
    # Step 3: Create integrated data flow
    print("\n🔄 STEP 3: Creating Integrated Training Data Flow")
    print("-" * 60)
    
    # Create data flow manager
    flow_manager = DataFlowManager()
    
    # Create integrated training flow
    training_flow = flow_manager.create_flow(
        "forklift_training_flow",
        "Forklift Training Integration Flow",
        "Integrates outline generation with standards lookup"
    )
    
    # Add nodes for processing
    def process_training_request(data):
        """Process training request and extract key information"""
        return {
            "prompt": data.get("prompt", ""),
            "topic_category": data.get("topic_category", ""),
            "learning_flow": data.get("learning_flow", []),
            "total_duration": data.get("total_duration", 0),
            "processed_at": "2024-01-01T00:00:00Z"
        }
    
    def filter_safety_topics(data):
        """Filter for safety-related topics"""
        return data.get("topic_category", "").endswith("_safety")
    
    def route_by_standard_type(data):
        """Route based on standard organization type"""
        standards = data.get("standards", [])
        if any(std.get("organization") == "OSHA" for std in standards):
            return "mandatory_compliance"
        else:
            return "recommended_compliance"
    
    # Create nodes
    process_node = TransformNode(
        "process_request",
        "Process Training Request",
        process_training_request
    )
    
    filter_node = FilterNode(
        "filter_safety",
        "Filter Safety Topics",
        filter_safety_topics
    )
    
    router_node = RouterNode(
        "route_compliance",
        "Route by Compliance Level",
        route_by_standard_type
    )
    router_node.add_route("mandatory_compliance", "mandatory_training")
    router_node.add_route("recommended_compliance", "recommended_training")
    
    # Add nodes to flow
    training_flow.add_node(process_node)
    training_flow.add_node(filter_node)
    training_flow.add_node(router_node)
    
    # Add connections
    training_flow.add_connection("process_request", "filter_safety")
    training_flow.add_connection("filter_safety", "route_compliance")
    
    print(f"Created flow: {training_flow.name}")
    print(f"Nodes: {list(training_flow.nodes.keys())}")
    print(f"Connections: {training_flow.connections}")
    
    # Step 4: Execute integrated workflow
    print("\n⚡ STEP 4: Executing Integrated Workflow")
    print("-" * 60)
    
    # Prepare integrated data
    integrated_data = {
        "prompt": user_prompt,
        "topic_category": training_outline["topic_category"],
        "learning_flow": training_outline["learning_flow"],
        "total_duration": training_outline["total_duration"],
        "standards": [
            {
                "id": std.id,
                "title": std.title,
                "standard_number": std.standard_number,
                "organization": std.organization.value,
                "compliance_level": std.compliance_level.value
            }
            for std in relevant_standards
        ]
    }
    
    # Execute flow
    flow_result = await training_flow.execute(integrated_data)
    
    print(f"Flow Execution Status: {flow_result['status']}")
    if flow_result['status'] == 'completed':
        print(f"Execution Time: {flow_result.get('execution_time', 'N/A')} seconds")
        
        # Show results from each node
        for node_id, result in flow_result['results'].items():
            if result['status'] == 'success':
                print(f"\n{node_id}: {result['payload']}")
    
    # Step 5: Generate comprehensive training report
    print("\n📊 STEP 5: Generating Comprehensive Training Report")
    print("-" * 60)
    
    training_report = {
        "training_request": {
            "prompt": user_prompt,
            "generated_at": "2024-01-01T00:00:00Z"
        },
        "training_outline": {
            "flow_summary": training_outline["flow_summary"],
            "total_duration": training_outline["total_duration"],
            "topics": training_outline["topics"]
        },
        "applicable_standards": [
            {
                "standard_number": std.standard_number,
                "title": std.title,
                "organization": std.organization.value,
                "compliance_level": std.compliance_level.value,
                "requirements": std.requirements[:3]  # First 3 requirements
            }
            for std in relevant_standards
        ],
        "compliance_summary": {
            "mandatory_standards": len([s for s in relevant_standards if s.compliance_level.value == "mandatory"]),
            "recommended_standards": len([s for s in relevant_standards if s.compliance_level.value == "recommended"]),
            "total_standards": len(relevant_standards)
        },
        "recommendations": [
            "Implement comprehensive forklift operator training program",
            "Ensure compliance with OSHA 1910.178 requirements",
            "Conduct regular safety inspections and maintenance",
            "Provide hands-on demonstration and assessment",
            "Maintain training records and certifications"
        ]
    }
    
    print("Training Report Generated:")
    print(f"  • Training Flow: {training_report['training_outline']['flow_summary']}")
    print(f"  • Total Duration: {training_report['training_outline']['total_duration']} minutes")
    print(f"  • Applicable Standards: {training_report['compliance_summary']['total_standards']}")
    print(f"  • Mandatory Standards: {training_report['compliance_summary']['mandatory_standards']}")
    print(f"  • Recommended Standards: {training_report['compliance_summary']['recommended_standards']}")
    
    print("\nKey Recommendations:")
    for i, rec in enumerate(training_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Step 6: Export results
    print("\n💾 STEP 6: Exporting Results")
    print("-" * 60)
    
    # Create output directory
    output_dir = Path("forklift_training_output")
    output_dir.mkdir(exist_ok=True)
    
    # Export training outline
    outline_yaml = outline_pipeline.export_outline("yaml")
    with open(output_dir / "forklift_training_outline.yaml", "w") as f:
        f.write(outline_yaml)
    
    # Export training report
    with open(output_dir / "forklift_training_report.json", "w") as f:
        json.dump(training_report, f, indent=2)
    
    # Export flow definition
    flow_def = training_flow.export_flow_definition("yaml")
    with open(output_dir / "forklift_training_flow.yaml", "w") as f:
        f.write(flow_def)
    
    print(f"Results exported to: {output_dir}")
    print("  • forklift_training_outline.yaml")
    print("  • forklift_training_report.json")
    print("  • forklift_training_flow.yaml")
    
    print("\n" + "=" * 80)
    print("✅ FORKLIFT TRAINING DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return training_report


async def demonstrate_other_safety_topics():
    """Demonstrate other safety training topics"""
    
    print("\n🔧 ADDITIONAL SAFETY TRAINING EXAMPLES")
    print("=" * 60)
    
    # Test different safety topics
    test_topics = [
        "Electrical safety hazards",
        "Machine guarding requirements", 
        "Fire safety procedures",
        "Personal protective equipment"
    ]
    
    outline_pipeline = OutlinePipeline()
    
    async with StandardsLookupPipeline() as standards_pipeline:
        for topic in test_topics:
            print(f"\n📋 Topic: '{topic}'")
            
            # Generate outline
            outline = outline_pipeline.create_safety_training_outline(topic)
            print(f"   Flow: {outline['flow_summary']}")
            print(f"   Duration: {outline['total_duration']} min")
            
            # Look up standards
            standards = standards_pipeline.lookup_workplace_safety_standards(topic)
            print(f"   Standards: {len(standards)} found")
            
            if standards:
                primary_standard = standards[0]
                print(f"   Primary: {primary_standard.standard_number} – {primary_standard.title}")


async def main():
    """Main demonstration function"""
    try:
        # Run main forklift demonstration
        await demonstrate_forklift_training()
        
        # Run additional examples
        await demonstrate_other_safety_topics()
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
