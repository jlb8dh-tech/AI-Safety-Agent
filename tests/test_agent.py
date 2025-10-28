"""
Tests for the Black Unicorn Agent

This module contains unit tests for all pipeline components.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Import components to test
from agent.pipelines.outline import (
    OutlinePipeline, Topic, TopicComplexity, TopicType, create_sample_topics
)
from agent.pipelines.standards import (
    StandardsLookupPipeline, Standard, StandardType, Jurisdiction, ComplianceLevel
)
from agent.pipelines.data_flows import (
    DataFlowManager, DataFlow, TransformNode, FilterNode, RouterNode, AggregatorNode
)
from agent.utils.helpers import (
    setup_logging, load_config, save_config, generate_id, extract_keywords,
    calculate_similarity, format_duration, create_sample_data
)


class TestOutlinePipeline:
    """Test cases for Outline Pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = OutlinePipeline()
        self.sample_topics = create_sample_topics()
    
    def test_add_topic(self):
        """Test adding topics to pipeline"""
        topic = self.sample_topics[0]
        result = self.pipeline.add_topic(topic)
        assert result is True
        assert topic.id in self.pipeline.topic_hierarchy.topic_details
    
    def test_build_hierarchy(self):
        """Test building topic hierarchy"""
        # Add sample topics
        for topic in self.sample_topics:
            self.pipeline.add_topic(topic)
        
        hierarchy = self.pipeline.build_hierarchy()
        assert len(hierarchy.root_topics) > 0
        assert len(hierarchy.topic_details) == len(self.sample_topics)
    
    def test_optimize_learning_path(self):
        """Test learning path optimization"""
        # Add sample topics
        for topic in self.sample_topics:
            self.pipeline.add_topic(topic)
        
        # Test path optimization
        path = self.pipeline.optimize_learning_path(["python_basics"])
        assert isinstance(path, list)
        assert len(path) > 0
    
    def test_validate_dependencies(self):
        """Test dependency validation"""
        # Add sample topics
        for topic in self.sample_topics:
            self.pipeline.add_topic(topic)
        
        issues = self.pipeline.validate_dependencies()
        assert isinstance(issues, list)
    
    def test_cluster_topics(self):
        """Test topic clustering"""
        # Add sample topics
        for topic in self.sample_topics:
            self.pipeline.add_topic(topic)
        
        clusters = self.pipeline.cluster_topics()
        assert isinstance(clusters, dict)
    
    def test_export_outline(self):
        """Test outline export"""
        # Add sample topics
        for topic in self.sample_topics:
            self.pipeline.add_topic(topic)
        
        # Test YAML export
        yaml_content = self.pipeline.export_outline("yaml")
        assert isinstance(yaml_content, str)
        assert "python_basics" in yaml_content
        
        # Test JSON export
        json_content = self.pipeline.export_outline("json")
        assert isinstance(json_content, str)


class TestStandardsLookupPipeline:
    """Test cases for Standards Lookup Pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.pipeline = StandardsLookupPipeline()
    
    def test_create_standard(self):
        """Test creating a standard object"""
        standard = Standard(
            id="test_standard",
            title="Test Standard",
            description="A test standard",
            standard_type=StandardType.REGULATION,
            compliance_level=ComplianceLevel.MANDATORY,
            jurisdiction=Jurisdiction.US_FEDERAL,
            version="1.0",
            effective_date=datetime.now()
        )
        
        assert standard.id == "test_standard"
        assert standard.title == "Test Standard"
        assert standard.standard_type == StandardType.REGULATION
    
    def test_search_standards(self):
        """Test standards search functionality"""
        # Add a test standard
        standard = Standard(
            id="test_standard",
            title="Test Standard",
            description="A test standard",
            standard_type=StandardType.REGULATION,
            compliance_level=ComplianceLevel.MANDATORY,
            jurisdiction=Jurisdiction.US_FEDERAL,
            version="1.0",
            effective_date=datetime.now(),
            keywords={"test", "standard"}
        )
        
        self.pipeline.standards_cache[standard.id] = standard
        
        # Test search
        results = self.pipeline.search_standards("test")
        assert len(results) > 0
        assert results[0].id == "test_standard"
    
    def test_get_compliance_report(self):
        """Test compliance report generation"""
        # Add test standards
        standard = Standard(
            id="test_standard",
            title="Test Standard",
            description="A test standard",
            standard_type=StandardType.REGULATION,
            compliance_level=ComplianceLevel.MANDATORY,
            jurisdiction=Jurisdiction.US_FEDERAL,
            version="1.0",
            effective_date=datetime.now(),
            categories={"privacy"}
        )
        
        self.pipeline.standards_cache[standard.id] = standard
        
        # Test compliance report
        org_profile = {
            "name": "Test Corp",
            "industry": "technology",
            "jurisdiction": "us_federal",
            "size": "large"
        }
        
        report = self.pipeline.get_compliance_report(org_profile)
        assert "organization" in report
        assert "applicable_standards" in report


class TestDataFlows:
    """Test cases for Data Flows"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = DataFlowManager()
    
    def test_create_flow(self):
        """Test creating a data flow"""
        flow = self.manager.create_flow("test_flow", "Test Flow", "A test flow")
        assert flow.flow_id == "test_flow"
        assert flow.name == "Test Flow"
    
    def test_add_transform_node(self):
        """Test adding transform node"""
        flow = self.manager.create_flow("test_flow", "Test Flow")
        
        transform_node = TransformNode(
            "transform_1",
            "Test Transform",
            lambda x: {"processed": x}
        )
        
        flow.add_node(transform_node)
        assert "transform_1" in flow.nodes
    
    def test_add_connection(self):
        """Test adding connections between nodes"""
        flow = self.manager.create_flow("test_flow", "Test Flow")
        
        # Add nodes
        node1 = TransformNode("node1", "Node 1", lambda x: x)
        node2 = TransformNode("node2", "Node 2", lambda x: x)
        
        flow.add_node(node1)
        flow.add_node(node2)
        
        # Add connection
        flow.add_connection("node1", "node2")
        assert ("node1", "node2") in flow.connections
    
    def test_validate_flow(self):
        """Test flow validation"""
        flow = self.manager.create_flow("test_flow", "Test Flow")
        
        # Add nodes
        node1 = TransformNode("node1", "Node 1", lambda x: x)
        node2 = TransformNode("node2", "Node 2", lambda x: x)
        
        flow.add_node(node1)
        flow.add_node(node2)
        flow.add_connection("node1", "node2")
        
        issues = flow.validate_flow()
        assert isinstance(issues, list)
    
    @pytest.mark.asyncio
    async def test_execute_flow(self):
        """Test flow execution"""
        flow = self.manager.create_flow("test_flow", "Test Flow")
        
        # Add transform node
        transform_node = TransformNode(
            "transform_1",
            "Test Transform",
            lambda x: {"processed": x, "timestamp": datetime.now().isoformat()}
        )
        
        flow.add_node(transform_node)
        
        # Execute flow
        result = await flow.execute({"test": "data"})
        assert result["status"] == "completed"
        assert "transform_1" in result["results"]


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    def test_generate_id(self):
        """Test ID generation"""
        id1 = generate_id("test")
        id2 = generate_id("test")
        
        assert id1.startswith("test_")
        assert id2.startswith("test_")
        assert id1 != id2  # Should be unique
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "This is a test document about Python programming and data analysis"
        keywords = extract_keywords(text)
        
        assert isinstance(keywords, list)
        assert "python" in keywords
        assert "programming" in keywords
        assert "data" in keywords
    
    def test_calculate_similarity(self):
        """Test similarity calculation"""
        text1 = "Python programming language"
        text2 = "Python coding language"
        text3 = "Java programming language"
        
        similarity1 = calculate_similarity(text1, text2)
        similarity2 = calculate_similarity(text1, text3)
        
        assert similarity1 > similarity2  # First pair should be more similar
        assert 0 <= similarity1 <= 1
        assert 0 <= similarity2 <= 1
    
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(30) == "30 minutes"
        assert format_duration(90) == "1 hour 30 minutes"
        assert format_duration(60) == "1 hour"
        assert format_duration(1500) == "1 day 1 hour"
    
    def test_config_operations(self):
        """Test configuration operations"""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {"test": "value", "nested": {"key": "value"}}
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading config
            loaded_config = load_config(config_path)
            assert loaded_config["test"] == "value"
            assert loaded_config["nested"]["key"] == "value"
            
            # Test saving config
            new_config = {"new": "data"}
            save_path = config_path.replace('.yaml', '_new.yaml')
            save_config(new_config, save_path)
            
            # Verify saved config
            saved_config = load_config(save_path)
            assert saved_config["new"] == "data"
            
        finally:
            # Clean up
            Path(config_path).unlink(missing_ok=True)
            Path(save_path).unlink(missing_ok=True)
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        sample_data = create_sample_data()
        
        assert "topics" in sample_data
        assert "organization" in sample_data
        assert len(sample_data["topics"]) > 0
        assert sample_data["organization"]["name"] == "Example Technology Corp"


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_pipeline_integration(self):
        """Test integration between pipelines"""
        # Create outline pipeline
        outline_pipeline = OutlinePipeline()
        
        # Add sample topics
        sample_topics = create_sample_topics()
        for topic in sample_topics:
            outline_pipeline.add_topic(topic)
        
        # Build hierarchy
        hierarchy = outline_pipeline.build_hierarchy()
        assert len(hierarchy.root_topics) > 0
        
        # Create data flow manager
        flow_manager = DataFlowManager()
        
        # Create integration flow
        flow = flow_manager.create_flow("integration_test", "Integration Test")
        
        # Add transform node
        transform_node = TransformNode(
            "process_topics",
            "Process Topics",
            lambda data: {"processed_topics": len(data.get("topics", []))}
        )
        
        flow.add_node(transform_node)
        
        # Execute flow
        test_data = {"topics": sample_topics}
        result = await flow.execute(test_data)
        
        assert result["status"] == "completed"
        assert "processed_topics" in result["results"]["process_topics"]["payload"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
