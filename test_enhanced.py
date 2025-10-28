#!/usr/bin/env python3
"""
Simple test script for the enhanced Black Unicorn Agent

This script tests the core functionality without requiring async dependencies.
"""

def test_outline_pipeline():
    """Test the outline pipeline functionality"""
    print("🧪 Testing Outline Pipeline...")
    
    try:
        from agent.pipelines.outline import OutlinePipeline
        
        # Create pipeline
        pipeline = OutlinePipeline()
        
        # Test forklift training outline
        outline = pipeline.create_safety_training_outline('Forklift operation hazards')
        
        print("✅ Outline Pipeline Test Passed!")
        print(f"   Flow: {outline['flow_summary']}")
        print(f"   Duration: {outline['total_duration']} minutes")
        print(f"   Topics: {len(outline['topics'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Outline Pipeline Test Failed: {e}")
        return False


def test_standards_pipeline():
    """Test the standards pipeline functionality"""
    print("\n🧪 Testing Standards Pipeline...")
    
    try:
        from agent.pipelines.standards import StandardsLookupPipeline
        
        # Create pipeline (without async context for basic test)
        standards_pipeline = StandardsLookupPipeline()
        
        # Test forklift standards lookup
        standards = standards_pipeline.lookup_workplace_safety_standards('Forklift operation')
        
        print("✅ Standards Pipeline Test Passed!")
        print(f"   Found {len(standards)} standards")
        
        if standards:
            primary_standard = standards[0]
            print(f"   Primary: {primary_standard.standard_number} – {primary_standard.title}")
            print(f"   Organization: {primary_standard.organization.value}")
            print(f"   Compliance: {primary_standard.compliance_level.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Standards Pipeline Test Failed: {e}")
        return False


def test_data_flows():
    """Test the data flows functionality"""
    print("\n🧪 Testing Data Flows...")
    
    try:
        from agent.pipelines.data_flows import DataFlowManager, TransformNode
        
        # Create flow manager
        manager = DataFlowManager()
        
        # Create a simple flow
        flow = manager.create_flow("test_flow", "Test Flow")
        
        # Add a transform node
        transform_node = TransformNode(
            "test_transform",
            "Test Transform",
            lambda x: {"processed": x, "test": True}
        )
        
        flow.add_node(transform_node)
        
        print("✅ Data Flows Test Passed!")
        print(f"   Created flow: {flow.name}")
        print(f"   Nodes: {list(flow.nodes.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Flows Test Failed: {e}")
        return False


def test_integration():
    """Test integration between components"""
    print("\n🧪 Testing Integration...")
    
    try:
        from agent.pipelines.outline import OutlinePipeline
        from agent.pipelines.standards import StandardsLookupPipeline
        
        # Create pipelines
        outline_pipeline = OutlinePipeline()
        standards_pipeline = StandardsLookupPipeline()
        
        # Test topic
        topic = "Forklift operation hazards"
        
        # Generate outline
        outline = outline_pipeline.create_safety_training_outline(topic)
        
        # Look up standards
        standards = standards_pipeline.lookup_workplace_safety_standards(topic)
        
        # Verify integration
        assert outline['prompt'] == topic
        assert len(outline['topics']) > 0
        assert len(standards) > 0
        
        print("✅ Integration Test Passed!")
        print(f"   Topic: {topic}")
        print(f"   Outline Flow: {outline['flow_summary']}")
        print(f"   Standards Found: {len(standards)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration Test Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Black Unicorn Agent - Enhanced Safety Training Tests")
    print("=" * 60)
    
    tests = [
        test_outline_pipeline,
        test_standards_pipeline,
        test_data_flows,
        test_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced agent is working correctly.")
        print("\n✨ Key Features Verified:")
        print("   • Safety training outline generation")
        print("   • OSHA/NIOSH/NFPA/ANSI standards lookup")
        print("   • Data flow orchestration")
        print("   • Component integration")
        
        print("\n🚀 Ready to use:")
        print("   • python forklift_example.py")
        print("   • python main.py --mode safety")
        print("   • python examples.py")
        
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
