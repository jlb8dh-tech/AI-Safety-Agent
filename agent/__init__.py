# Black Unicorn Agent Package

"""
Black Unicorn Agent - A modular agent system with specialized pipelines for:
- Training topic structuring (Outline Pipeline)
- Regulatory standards lookup (Standards Lookup Pipeline)  
- Clean data flows connecting agent workflows (Data Flows)

This package provides the core components for building intelligent agent workflows
that can process training content and ensure regulatory compliance.
"""

__version__ = "1.0.0"
__author__ = "Black Unicorn Team"
__description__ = "Modular agent system with training and compliance pipelines"

# Import main components
from .pipelines.outline import OutlinePipeline, Topic, TopicComplexity, TopicType
from .pipelines.standards import StandardsLookupPipeline, Standard, StandardType, Jurisdiction
from .pipelines.data_flows import DataFlowManager, DataFlow, FlowNode, TransformNode, FilterNode

__all__ = [
    "OutlinePipeline",
    "Topic", 
    "TopicComplexity",
    "TopicType",
    "StandardsLookupPipeline",
    "Standard",
    "StandardType", 
    "Jurisdiction",
    "DataFlowManager",
    "DataFlow",
    "FlowNode",
    "TransformNode", 
    "FilterNode"
]
