#!/usr/bin/env python3
"""
Example usage of the Black Unicorn Agent

This script demonstrates how to use the agent's pipeline components
for training topic structuring and compliance checking.
"""

import asyncio
import json
from pathlib import Path
from agent.pipelines.outline import OutlinePipeline, Topic, TopicComplexity, TopicType
from agent.pipelines.standards import StandardsLookupPipeline, Jurisdiction
from agent.pipelines.data_flows import DataFlowManager, TransformNode, FilterNode
from agent.utils.helpers import create_sample_data, setup_logging


async def example_outline_pipeline():
    """Example of using the Outline Pipeline"""
    print("=== Outline Pipeline Example ===")
    
    # Create pipeline
    pipeline = OutlinePipeline()
    
    # Create sample topics
    topics = [
        Topic(
            id="python_basics",
            title="Python Programming Basics",
            description="Introduction to Python programming language",
            complexity=TopicComplexity.BEGINNER,
            topic_type=TopicType.CONCEPTUAL,
            estimated_duration=90,
            learning_objectives=["Learn Python syntax", "Write basic programs"],
            keywords={"python", "programming", "basics"}
        ),
        Topic(
            id="data_structures",
            title="Data Structures in Python",
            description="Understanding lists, dictionaries, and other data structures",
            complexity=TopicComplexity.INTERMEDIATE,
            topic_type=TopicType.THEORETICAL,
            estimated_duration=120,
            prerequisites={"python_basics"},
            learning_objectives=["Master data structures", "Choose appropriate structures"],
            keywords={"data structures", "lists", "dictionaries"}
        ),
        Topic(
            id="data_privacy",
            title="Data Privacy Fundamentals",
            description="Understanding data privacy regulations and best practices",
            complexity=TopicComplexity.INTERMEDIATE,
            topic_type=TopicType.THEORETICAL,
            estimated_duration=150,
            learning_objectives=["Understand privacy laws", "Implement privacy controls"],
            keywords={"privacy", "gdpr", "data protection"}
        )
    ]
    
    # Add topics to pipeline
    for topic in topics:
        pipeline.add_topic(topic)
        print(f"Added topic: {topic.title}")
    
    # Build hierarchy
    hierarchy = pipeline.build_hierarchy()
    print(f"\nHierarchy built with {len(hierarchy.root_topics)} root topics")
    
    # Generate learning paths
    for root_topic in hierarchy.root_topics:
        path = pipeline.optimize_learning_path([root_topic])
        print(f"Learning path from {root_topic}: {path}")
    
    # Validate dependencies
    issues = pipeline.validate_dependencies()
    if issues:
        print(f"\nDependency issues found: {issues}")
    else:
        print("\nNo dependency issues found")
    
    # Export outline
    outline_yaml = pipeline.export_outline("yaml")
    print(f"\nExported outline ({len(outline_yaml)} characters)")
    
    return pipeline


async def example_standards_pipeline():
    """Example of using the Standards Lookup Pipeline"""
    print("\n=== Standards Lookup Pipeline Example ===")
    
    # Create pipeline
    async with StandardsLookupPipeline() as pipeline:
        # Create sample standards
        from agent.pipelines.standards import Standard, StandardType, ComplianceLevel
        
        sample_standards = [
            Standard(
                id="gdpr_sample",
                title="General Data Protection Regulation",
                description="EU regulation on data protection and privacy",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.EU,
                version="1.0",
                effective_date="2018-05-25",
                keywords={"privacy", "data protection", "gdpr", "eu"},
                categories={"privacy", "data protection"}
            ),
            Standard(
                id="ccpa_sample",
                title="California Consumer Privacy Act",
                description="California state law on consumer privacy rights",
                standard_type=StandardType.REGULATION,
                compliance_level=ComplianceLevel.MANDATORY,
                jurisdiction=Jurisdiction.US_STATE,
                version="1.0",
                effective_date="2020-01-01",
                keywords={"privacy", "ccpa", "california", "consumer rights"},
                categories={"privacy", "consumer protection"}
            )
        ]
        
        # Add to cache
        for standard in sample_standards:
            pipeline.standards_cache[standard.id] = standard
            print(f"Added standard: {standard.title}")
        
        # Search for standards
        search_results = pipeline.search_standards("privacy", limit=5)
        print(f"\nFound {len(search_results)} privacy-related standards")
        
        # Generate compliance report
        org_profile = {
            "name": "Example Corp",
            "industry": "technology",
            "jurisdiction": "us_federal",
            "size": "large"
        }
        
        report = pipeline.get_compliance_report(org_profile)
        print(f"\nCompliance report generated for {report['organization']}")
        print(f"Applicable standards: {len(report['applicable_standards'])}")
        
        return pipeline


async def example_data_flows():
    """Example of using Data Flows"""
    print("\n=== Data Flows Example ===")
    
    # Create flow manager
    manager = DataFlowManager()
    
    # Create a flow
    flow = manager.create_flow(
        "example_flow",
        "Example Data Processing Flow",
        "Demonstrates data transformation and filtering"
    )
    
    # Create nodes
    transform_node = TransformNode(
        "transform_data",
        "Data Transformer",
        lambda x: {
            "original": x,
            "processed": True,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )
    
    filter_node = FilterNode(
        "filter_data",
        "Data Filter",
        lambda x: isinstance(x, dict) and x.get("processed", False)
    )
    
    # Add nodes to flow
    flow.add_node(transform_node)
    flow.add_node(filter_node)
    
    # Add connection
    flow.add_connection("transform_data", "filter_data")
    
    print(f"Created flow: {flow.name}")
    print(f"Nodes: {list(flow.nodes.keys())}")
    print(f"Connections: {flow.connections}")
    
    # Validate flow
    issues = flow.validate_flow()
    if issues:
        print(f"Flow validation issues: {issues}")
    else:
        print("Flow validation passed")
    
    # Execute flow
    test_data = {"message": "Hello World"}
    result = await flow.execute(test_data)
    
    print(f"\nFlow execution result:")
    print(f"Status: {result['status']}")
    print(f"Execution time: {result.get('execution_time', 'N/A')} seconds")
    
    return manager


async def example_integration():
    """Example of integrated workflow"""
    print("\n=== Integrated Workflow Example ===")
    
    # Create sample data
    sample_data = create_sample_data()
    
    # Create outline pipeline
    outline_pipeline = OutlinePipeline()
    
    # Add topics from sample data
    for topic_data in sample_data["topics"]:
        topic = Topic(
            id=topic_data["id"],
            title=topic_data["title"],
            description=topic_data["description"],
            complexity=TopicComplexity(topic_data["complexity"]),
            topic_type=TopicType(topic_data["type"]),
            estimated_duration=topic_data["duration"],
            learning_objectives=topic_data["objectives"],
            keywords=set(topic_data["keywords"])
        )
        outline_pipeline.add_topic(topic)
    
    # Build hierarchy
    hierarchy = outline_pipeline.build_hierarchy()
    print(f"Built outline with {len(hierarchy.topic_details)} topics")
    
    # Create data flow for processing
    manager = DataFlowManager()
    flow = manager.create_flow("integration_flow", "Integration Flow")
    
    # Add processing node
    def process_topics(data):
        return {
            "topic_count": len(data.get("topics", [])),
            "processed_at": "2024-01-01T00:00:00Z",
            "status": "processed"
        }
    
    process_node = TransformNode("process_topics", "Process Topics", process_topics)
    flow.add_node(process_node)
    
    # Execute flow
    result = await flow.execute(sample_data)
    print(f"Integration flow executed: {result['status']}")
    
    return {
        "outline": outline_pipeline,
        "flow_manager": manager,
        "sample_data": sample_data
    }


async def main():
    """Main example function"""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting Black Unicorn Agent examples")
    
    try:
        # Run examples
        await example_outline_pipeline()
        await example_standards_pipeline()
        await example_data_flows()
        integration_result = await example_integration()
        
        print("\n=== All Examples Completed Successfully ===")
        
        # Export results
        output_dir = Path("examples_output")
        output_dir.mkdir(exist_ok=True)
        
        # Export outline
        outline_yaml = integration_result["outline"].export_outline("yaml")
        with open(output_dir / "example_outline.yaml", "w") as f:
            f.write(outline_yaml)
        
        # Export sample data
        with open(output_dir / "sample_data.json", "w") as f:
            json.dump(integration_result["sample_data"], f, indent=2)
        
        print(f"Results exported to {output_dir}")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
