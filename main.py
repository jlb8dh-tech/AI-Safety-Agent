"""
Main entry point for the Black Unicorn Agent

This module provides the main interface for running the agent and integrating
all pipeline components.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import argparse
from datetime import datetime

# Import pipeline components
from agent.pipelines.outline import OutlinePipeline, Topic, TopicComplexity, TopicType, create_sample_topics
from agent.pipelines.standards import StandardsLookupPipeline, Jurisdiction, StandardType
from agent.pipelines.data_flows import DataFlowManager, TransformNode, FilterNode, RouterNode, AggregatorNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/agent.log')
    ]
)
logger = logging.getLogger(__name__)


class BlackUnicornAgent:
    """
    Main agent class that orchestrates all pipeline components
    
    Features:
    - Integrated pipeline management
    - Workflow orchestration
    - Configuration management
    - Error handling and monitoring
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the agent"""
        self.config = self._load_config(config_path)
        self.outline_pipeline = OutlinePipeline(self.config.get('outline_pipeline', {}))
        self.standards_pipeline = None  # Will be initialized in async context
        self.data_flow_manager = DataFlowManager()
        self.initialized = False
        
        # Create necessary directories
        self._setup_directories()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Use default config
        default_config_path = Path(__file__).parent / "config" / "config.example.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {}
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'data', 'exports']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize all pipeline components"""
        try:
            logger.info("Initializing Black Unicorn Agent...")
            
            # Initialize standards pipeline with async context
            self.standards_pipeline = StandardsLookupPipeline(self.config.get('standards_lookup', {}))
            
            # Create default data flows
            await self._create_default_flows()
            
            self.initialized = True
            logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def _create_default_flows(self):
        """Create default data flows for pipeline integration"""
        
        # Flow 1: Outline to Standards Integration
        outline_to_standards_flow = self.data_flow_manager.create_flow(
            "outline_to_standards",
            "Outline to Standards Integration",
            "Maps training topics to relevant regulatory standards"
        )
        
        # Transform node: Extract keywords from topic
        def extract_topic_keywords(data):
            if isinstance(data, dict) and 'topic' in data:
                topic = data['topic']
                return {
                    'keywords': list(topic.keywords) if hasattr(topic, 'keywords') else [],
                    'categories': list(topic.categories) if hasattr(topic, 'categories') else [],
                    'topic_id': topic.id if hasattr(topic, 'id') else None
                }
            return {'keywords': [], 'categories': [], 'topic_id': None}
        
        transform_node = TransformNode(
            "extract_keywords",
            "Extract Topic Keywords",
            extract_topic_keywords
        )
        
        # Filter node: Filter out topics without keywords
        filter_node = FilterNode(
            "filter_topics",
            "Filter Topics with Keywords",
            lambda data: data.get('keywords', [])
        )
        
        # Router node: Route based on topic categories
        def route_by_category(data):
            categories = data.get('categories', [])
            if 'privacy' in categories:
                return 'privacy_standards'
            elif 'security' in categories:
                return 'security_standards'
            else:
                return 'general_standards'
        
        router_node = RouterNode(
            "route_by_category",
            "Route by Topic Category",
            route_by_category
        )
        router_node.add_route('privacy_standards', 'privacy_pipeline')
        router_node.add_route('security_standards', 'security_pipeline')
        router_node.add_route('general_standards', 'general_pipeline')
        
        # Add nodes to flow
        outline_to_standards_flow.add_node(transform_node)
        outline_to_standards_flow.add_node(filter_node)
        outline_to_standards_flow.add_node(router_node)
        
        # Add connections
        outline_to_standards_flow.add_connection("extract_keywords", "filter_topics")
        outline_to_standards_flow.add_connection("filter_topics", "route_by_category")
        
        # Flow 2: Standards Compliance Check
        compliance_check_flow = self.data_flow_manager.create_flow(
            "compliance_check",
            "Standards Compliance Check",
            "Checks compliance status against relevant standards"
        )
        
        # Transform node: Prepare compliance data
        def prepare_compliance_data(data):
            return {
                'organization_profile': data.get('organization', {}),
                'standards_to_check': data.get('standards', []),
                'check_timestamp': datetime.now().isoformat()
            }
        
        compliance_transform = TransformNode(
            "prepare_compliance",
            "Prepare Compliance Data",
            prepare_compliance_data
        )
        
        # Aggregator node: Aggregate compliance results
        def aggregate_compliance_results(results):
            total_checks = len(results)
            passed_checks = sum(1 for result in results if result.get('compliant', False))
            compliance_rate = passed_checks / total_checks if total_checks > 0 else 0
            
            return {
                'total_standards': total_checks,
                'compliant_standards': passed_checks,
                'compliance_rate': compliance_rate,
                'non_compliant': [r for r in results if not r.get('compliant', False)],
                'summary': f"Compliance rate: {compliance_rate:.2%}"
            }
        
        compliance_aggregator = AggregatorNode(
            "aggregate_results",
            "Aggregate Compliance Results",
            aggregate_compliance_results
        )
        compliance_aggregator.buffer_size = 10
        
        # Add nodes to compliance flow
        compliance_check_flow.add_node(compliance_transform)
        compliance_check_flow.add_node(compliance_aggregator)
        
        # Add connection
        compliance_check_flow.add_connection("prepare_compliance", "aggregate_results")
        
        logger.info("Default data flows created successfully")
    
    async def create_safety_training_outline(self, topic_prompt: str) -> Dict[str, Any]:
        """
        Create a safety training outline from a user prompt
        
        Args:
            topic_prompt: User prompt describing the safety topic
            
        Returns:
            Dict containing structured training outline
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info(f"Creating safety training outline for: {topic_prompt}")
            
            # Use the enhanced outline pipeline
            training_outline = self.outline_pipeline.create_safety_training_outline(topic_prompt)
            
            # Look up relevant standards
            relevant_standards = []
            if self.standards_pipeline:
                async with self.standards_pipeline:
                    relevant_standards = self.standards_pipeline.lookup_workplace_safety_standards(topic_prompt)
            
            result = {
                'status': 'success',
                'prompt': topic_prompt,
                'training_outline': training_outline,
                'relevant_standards': [
                    {
                        'standard_number': std.standard_number,
                        'title': std.title,
                        'organization': std.organization.value,
                        'compliance_level': std.compliance_level.value,
                        'requirements': std.requirements[:3]  # First 3 requirements
                    }
                    for std in relevant_standards
                ],
                'compliance_summary': {
                    'mandatory_standards': len([s for s in relevant_standards if s.compliance_level.value == "mandatory"]),
                    'recommended_standards': len([s for s in relevant_standards if s.compliance_level.value == "recommended"]),
                    'total_standards': len(relevant_standards)
                }
            }
            
            logger.info(f"Safety training outline created successfully: {training_outline['flow_summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Error creating safety training outline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def process_training_outline(self, topics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process training topics and create structured outline
        
        Args:
            topics_data: Dictionary containing topic information
            
        Returns:
            Dict containing processed outline and related standards
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info("Processing training outline...")
            
            # Add topics to outline pipeline
            topics_added = 0
            for topic_data in topics_data.get('topics', []):
                topic = Topic(
                    id=topic_data['id'],
                    title=topic_data['title'],
                    description=topic_data['description'],
                    complexity=TopicComplexity(topic_data.get('complexity', 'beginner')),
                    topic_type=TopicType(topic_data.get('type', 'conceptual')),
                    estimated_duration=topic_data.get('duration', 60),
                    prerequisites=set(topic_data.get('prerequisites', [])),
                    learning_objectives=topic_data.get('objectives', []),
                    keywords=set(topic_data.get('keywords', [])),
                    resources=topic_data.get('resources', []),
                    assessment_criteria=topic_data.get('assessment', [])
                )
                
                if self.outline_pipeline.add_topic(topic):
                    topics_added += 1
            
            # Build hierarchy
            hierarchy = self.outline_pipeline.build_hierarchy()
            
            # Validate dependencies
            issues = self.outline_pipeline.validate_dependencies()
            
            # Find relevant standards for topics
            relevant_standards = []
            if self.standards_pipeline:
                async with self.standards_pipeline:
                    for topic_id, topic in self.outline_pipeline.topic_hierarchy.topic_details.items():
                        # Search for standards related to topic keywords
                        standards = self.standards_pipeline.search_standards(
                            " ".join(topic.keywords), limit=5
                        )
                        relevant_standards.extend([
                            {
                                'topic_id': topic_id,
                                'standard_id': std.id,
                                'title': std.title,
                                'compliance_level': std.compliance_level.value,
                                'jurisdiction': std.jurisdiction.value
                            }
                            for std in standards
                        ])
            
            result = {
                'status': 'success',
                'topics_added': topics_added,
                'hierarchy': {
                    'root_topics': hierarchy.root_topics,
                    'total_topics': len(hierarchy.topic_details),
                    'topic_graph': hierarchy.topic_graph
                },
                'validation_issues': issues,
                'relevant_standards': relevant_standards,
                'learning_paths': self._generate_learning_paths(hierarchy)
            }
            
            logger.info(f"Training outline processed successfully: {topics_added} topics added")
            return result
            
        except Exception as e:
            logger.error(f"Error processing training outline: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_learning_paths(self, hierarchy) -> Dict[str, Any]:
        """Generate optimized learning paths"""
        paths = {}
        
        for root_topic in hierarchy.root_topics:
            try:
                learning_path = self.outline_pipeline.optimize_learning_path([root_topic])
                paths[root_topic] = {
                    'path': learning_path,
                    'estimated_duration': sum(
                        self.outline_pipeline.topic_hierarchy.topic_details[topic_id].estimated_duration
                        for topic_id in learning_path
                        if topic_id in self.outline_pipeline.topic_hierarchy.topic_details
                    )
                }
            except Exception as e:
                logger.warning(f"Could not generate learning path for {root_topic}: {e}")
                paths[root_topic] = {'path': [], 'estimated_duration': 0}
        
        return paths
    
    async def check_compliance(self, organization_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check organization compliance against relevant standards
        
        Args:
            organization_profile: Organization details and characteristics
            
        Returns:
            Dict containing compliance report
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info("Checking compliance...")
            
            if not self.standards_pipeline:
                return {
                    'status': 'error',
                    'error': 'Standards pipeline not available'
                }
            
            async with self.standards_pipeline:
                # Fetch relevant standards
                standards = await self.standards_pipeline.fetch_standards(
                    jurisdiction=Jurisdiction(organization_profile.get('jurisdiction', 'us_federal'))
                )
                
                # Generate compliance report
                report = self.standards_pipeline.get_compliance_report(organization_profile)
                
                # Execute compliance check flow
                compliance_data = {
                    'organization': organization_profile,
                    'standards': standards
                }
                
                flow_result = await self.data_flow_manager.execute_flow(
                    "compliance_check", compliance_data
                )
                
                result = {
                    'status': 'success',
                    'compliance_report': report,
                    'flow_execution': flow_result,
                    'standards_checked': len(standards)
                }
                
                logger.info("Compliance check completed successfully")
                return result
                
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def run_integrated_workflow(self, 
                                    topics_data: Dict[str, Any],
                                    organization_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run integrated workflow combining outline and compliance checking
        
        Args:
            topics_data: Training topics data
            organization_profile: Organization profile for compliance
            
        Returns:
            Dict containing integrated results
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            logger.info("Running integrated workflow...")
            
            # Process training outline
            outline_result = await self.process_training_outline(topics_data)
            
            # Check compliance
            compliance_result = await self.check_compliance(organization_profile)
            
            # Execute outline to standards integration flow
            integration_data = {
                'topics': topics_data.get('topics', []),
                'standards': compliance_result.get('compliance_report', {}).get('applicable_standards', [])
            }
            
            integration_result = await self.data_flow_manager.execute_flow(
                "outline_to_standards", integration_data
            )
            
            result = {
                'status': 'success',
                'outline_processing': outline_result,
                'compliance_check': compliance_result,
                'integration_flow': integration_result,
                'workflow_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Integrated workflow completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in integrated workflow: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def export_results(self, results: Dict[str, Any], format: str = "yaml", output_path: Optional[str] = None) -> str:
        """
        Export results to file
        
        Args:
            results: Results to export
            format: Export format ("yaml", "json")
            output_path: Optional output file path
            
        Returns:
            str: Exported content
        """
        try:
            if format.lower() == "yaml":
                content = yaml.dump(results, default_flow_style=False, sort_keys=False)
            elif format.lower() == "json":
                import json
                content = json.dumps(results, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if output_path:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write(content)
                logger.info(f"Results exported to {output_path}")
            
            return content
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Black Unicorn Agent")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--mode", choices=["outline", "compliance", "integrated", "safety"], 
                       default="integrated", help="Operation mode")
    parser.add_argument("--input", help="Path to input data file")
    parser.add_argument("--output", help="Path to output file")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = BlackUnicornAgent(args.config)
    await agent.initialize()
    
    # Load input data if provided
    input_data = {}
    if args.input and Path(args.input).exists():
        with open(args.input, 'r') as f:
            if args.input.endswith('.yaml') or args.input.endswith('.yml'):
                input_data = yaml.safe_load(f)
            elif args.input.endswith('.json'):
                import json
                input_data = json.load(f)
    
    # Default sample data if no input provided
    if not input_data:
        input_data = {
            'topics': [
                {
                    'id': 'python_basics',
                    'title': 'Python Basics',
                    'description': 'Introduction to Python programming',
                    'complexity': 'beginner',
                    'type': 'conceptual',
                    'duration': 90,
                    'keywords': ['python', 'programming', 'basics'],
                    'objectives': ['Learn Python syntax', 'Write basic programs']
                },
                {
                    'id': 'data_privacy',
                    'title': 'Data Privacy Fundamentals',
                    'description': 'Understanding data privacy regulations',
                    'complexity': 'intermediate',
                    'type': 'theoretical',
                    'duration': 120,
                    'keywords': ['privacy', 'gdpr', 'data protection'],
                    'objectives': ['Understand privacy laws', 'Implement privacy controls']
                }
            ],
            'organization': {
                'name': 'Example Corp',
                'industry': 'technology',
                'jurisdiction': 'us_federal',
                'size': 'large'
            }
        }
    
    # Execute based on mode
    if args.mode == "outline":
        result = await agent.process_training_outline(input_data)
    elif args.mode == "compliance":
        result = await agent.check_compliance(input_data.get('organization', {}))
    elif args.mode == "safety":
        # For safety mode, use the prompt from input data or default
        safety_prompt = input_data.get('safety_prompt', 'Forklift operation hazards')
        result = await agent.create_safety_training_outline(safety_prompt)
    else:  # integrated
        result = await agent.run_integrated_workflow(
            input_data, 
            input_data.get('organization', {})
        )
    
    # Export results
    exported_content = agent.export_results(result, args.format, args.output)
    
    if not args.output:
        print("Results:")
        print(exported_content)


if __name__ == "__main__":
    asyncio.run(main())
