"""
Data Flows - Clean, well-documented data flows that connect to agent workflows

This module provides standardized data flow patterns, validation, and transformation
utilities for connecting different agent workflows and pipelines.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid
from pathlib import Path
from pydantic import BaseModel, Field, validator
import yaml
import networkx as nx
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class FlowStatus(Enum):
    """Enumeration for flow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataType(Enum):
    """Enumeration for supported data types"""
    JSON = "json"
    YAML = "yaml"
    TEXT = "text"
    BINARY = "binary"
    CSV = "csv"
    XML = "xml"


class ValidationLevel(Enum):
    """Enumeration for validation levels"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    CUSTOM = "custom"


@dataclass
class DataPacket:
    """Represents a data packet flowing through the system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    data_type: DataType = DataType.JSON
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    destination: Optional[str] = None
    validation_status: bool = False
    error_message: Optional[str] = None


class FlowNode(ABC):
    """Abstract base class for flow nodes"""
    
    def __init__(self, node_id: str, name: str, description: str = ""):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.input_schema: Optional[Dict] = None
        self.output_schema: Optional[Dict] = None
        self.validation_level = ValidationLevel.BASIC
    
    @abstractmethod
    async def process(self, data_packet: DataPacket) -> DataPacket:
        """Process a data packet"""
        pass
    
    def validate_input(self, data_packet: DataPacket) -> bool:
        """Validate input data packet"""
        if self.validation_level == ValidationLevel.NONE:
            return True
        
        if not data_packet.payload:
            return False
        
        if self.validation_level == ValidationLevel.BASIC:
            return isinstance(data_packet.payload, (dict, list, str, int, float, bool))
        
        if self.validation_level == ValidationLevel.STRICT and self.input_schema:
            # In a real implementation, you'd use a schema validation library like jsonschema
            return True
        
        return True
    
    def validate_output(self, data_packet: DataPacket) -> bool:
        """Validate output data packet"""
        if self.validation_level == ValidationLevel.NONE:
            return True
        
        if not data_packet.payload:
            return False
        
        if self.validation_level == ValidationLevel.BASIC:
            return isinstance(data_packet.payload, (dict, list, str, int, float, bool))
        
        if self.validation_level == ValidationLevel.STRICT and self.output_schema:
            # In a real implementation, you'd use a schema validation library like jsonschema
            return True
        
        return True


class TransformNode(FlowNode):
    """Node that transforms data"""
    
    def __init__(self, node_id: str, name: str, transform_func: Callable[[Any], Any], **kwargs):
        super().__init__(node_id, name, **kwargs)
        self.transform_func = transform_func
    
    async def process(self, data_packet: DataPacket) -> DataPacket:
        """Transform the data packet"""
        try:
            if not self.validate_input(data_packet):
                data_packet.error_message = "Input validation failed"
                return data_packet
            
            # Apply transformation
            transformed_data = self.transform_func(data_packet.payload)
            
            # Create new packet with transformed data
            output_packet = DataPacket(
                data_type=data_packet.data_type,
                payload=transformed_data,
                metadata=data_packet.metadata.copy(),
                source=self.node_id,
                destination=data_packet.destination
            )
            
            # Validate output
            if not self.validate_output(output_packet):
                output_packet.error_message = "Output validation failed"
                return output_packet
            
            output_packet.validation_status = True
            return output_packet
            
        except Exception as e:
            logger.error(f"Error in transform node {self.node_id}: {e}")
            data_packet.error_message = str(e)
            return data_packet


class FilterNode(FlowNode):
    """Node that filters data based on conditions"""
    
    def __init__(self, node_id: str, name: str, filter_func: Callable[[Any], bool], **kwargs):
        super().__init__(node_id, name, **kwargs)
        self.filter_func = filter_func
    
    async def process(self, data_packet: DataPacket) -> DataPacket:
        """Filter the data packet"""
        try:
            if not self.validate_input(data_packet):
                data_packet.error_message = "Input validation failed"
                return data_packet
            
            # Apply filter
            if self.filter_func(data_packet.payload):
                data_packet.validation_status = True
                return data_packet
            else:
                # Filter out the packet
                data_packet.error_message = "Packet filtered out"
                return data_packet
            
        except Exception as e:
            logger.error(f"Error in filter node {self.node_id}: {e}")
            data_packet.error_message = str(e)
            return data_packet


class RouterNode(FlowNode):
    """Node that routes data to different outputs based on conditions"""
    
    def __init__(self, node_id: str, name: str, route_func: Callable[[Any], str], **kwargs):
        super().__init__(node_id, name, **kwargs)
        self.route_func = route_func
        self.routes: Dict[str, str] = {}
    
    def add_route(self, condition: str, destination: str):
        """Add a routing condition"""
        self.routes[condition] = destination
    
    async def process(self, data_packet: DataPacket) -> DataPacket:
        """Route the data packet"""
        try:
            if not self.validate_input(data_packet):
                data_packet.error_message = "Input validation failed"
                return data_packet
            
            # Determine route
            route = self.route_func(data_packet.payload)
            if route in self.routes:
                data_packet.destination = self.routes[route]
            else:
                data_packet.error_message = f"No route found for condition: {route}"
                return data_packet
            
            data_packet.validation_status = True
            return data_packet
            
        except Exception as e:
            logger.error(f"Error in router node {self.node_id}: {e}")
            data_packet.error_message = str(e)
            return data_packet


class AggregatorNode(FlowNode):
    """Node that aggregates multiple data packets"""
    
    def __init__(self, node_id: str, name: str, aggregation_func: Callable[[List[Any]], Any], **kwargs):
        super().__init__(node_id, name, **kwargs)
        self.aggregation_func = aggregation_func
        self.buffer: List[DataPacket] = []
        self.buffer_size: int = kwargs.get('buffer_size', 10)
        self.timeout: float = kwargs.get('timeout', 30.0)
    
    async def process(self, data_packet: DataPacket) -> DataPacket:
        """Aggregate data packets"""
        try:
            if not self.validate_input(data_packet):
                data_packet.error_message = "Input validation failed"
                return data_packet
            
            # Add to buffer
            self.buffer.append(data_packet)
            
            # Check if buffer is full or timeout reached
            if len(self.buffer) >= self.buffer_size:
                return await self._flush_buffer()
            
            # Return a placeholder packet indicating buffering
            placeholder = DataPacket(
                payload={"status": "buffering", "count": len(self.buffer)},
                metadata={"aggregator": self.node_id},
                source=self.node_id
            )
            return placeholder
            
        except Exception as e:
            logger.error(f"Error in aggregator node {self.node_id}: {e}")
            data_packet.error_message = str(e)
            return data_packet
    
    async def _flush_buffer(self) -> DataPacket:
        """Flush the buffer and create aggregated packet"""
        try:
            # Extract payloads
            payloads = [packet.payload for packet in self.buffer if packet.payload is not None]
            
            # Apply aggregation function
            aggregated_payload = self.aggregation_func(payloads)
            
            # Create aggregated packet
            aggregated_packet = DataPacket(
                data_type=DataType.JSON,
                payload=aggregated_payload,
                metadata={
                    "aggregated_from": len(self.buffer),
                    "aggregator": self.node_id,
                    "timestamp": datetime.now().isoformat()
                },
                source=self.node_id
            )
            
            # Clear buffer
            self.buffer.clear()
            
            aggregated_packet.validation_status = True
            return aggregated_packet
            
        except Exception as e:
            logger.error(f"Error flushing aggregator buffer: {e}")
            error_packet = DataPacket(
                payload=None,
                error_message=str(e),
                source=self.node_id
            )
            return error_packet


class DataFlow:
    """Represents a complete data flow with nodes and connections"""
    
    def __init__(self, flow_id: str, name: str, description: str = ""):
        self.flow_id = flow_id
        self.name = name
        self.description = description
        self.nodes: Dict[str, FlowNode] = {}
        self.connections: List[tuple] = []  # (source_node_id, target_node_id)
        self.graph = nx.DiGraph()
        self.status = FlowStatus.PENDING
        self.execution_history: List[Dict] = []
        self.created_at = datetime.now()
        self.last_executed = None
    
    def add_node(self, node: FlowNode) -> None:
        """Add a node to the flow"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, node=node)
    
    def add_connection(self, source_node_id: str, target_node_id: str) -> None:
        """Add a connection between nodes"""
        if source_node_id in self.nodes and target_node_id in self.nodes:
            self.connections.append((source_node_id, target_node_id))
            self.graph.add_edge(source_node_id, target_node_id)
        else:
            raise ValueError(f"One or both nodes not found: {source_node_id}, {target_node_id}")
    
    def validate_flow(self) -> List[str]:
        """Validate the flow structure"""
        issues = []
        
        # Check for cycles
        try:
            nx.topological_sort(self.graph)
        except nx.NetworkXError:
            cycles = list(nx.simple_cycles(self.graph))
            issues.append(f"Flow contains cycles: {cycles}")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            issues.append(f"Isolated nodes found: {isolated_nodes}")
        
        # Check for unreachable nodes
        if self.graph.nodes():
            reachable = set()
            for node in self.graph.nodes():
                reachable.update(nx.descendants(self.graph, node))
                reachable.add(node)
            
            unreachable = set(self.graph.nodes()) - reachable
            if unreachable:
                issues.append(f"Unreachable nodes: {unreachable}")
        
        return issues
    
    async def execute(self, input_data: Any, start_node_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the data flow"""
        self.status = FlowStatus.RUNNING
        execution_id = str(uuid.uuid4())
        
        try:
            # Validate flow before execution
            issues = self.validate_flow()
            if issues:
                raise ValueError(f"Flow validation failed: {issues}")
            
            # Determine start node
            if start_node_id:
                if start_node_id not in self.nodes:
                    raise ValueError(f"Start node not found: {start_node_id}")
                start_nodes = [start_node_id]
            else:
                # Find nodes with no incoming edges
                start_nodes = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            
            if not start_nodes:
                raise ValueError("No start nodes found")
            
            # Create initial data packet
            initial_packet = DataPacket(
                payload=input_data,
                source="external",
                data_type=DataType.JSON
            )
            
            # Execute flow
            results = await self._execute_from_nodes(start_nodes, initial_packet)
            
            self.status = FlowStatus.COMPLETED
            self.last_executed = datetime.now()
            
            execution_record = {
                "execution_id": execution_id,
                "status": self.status.value,
                "start_time": self.created_at.isoformat(),
                "end_time": self.last_executed.isoformat(),
                "results": results
            }
            self.execution_history.append(execution_record)
            
            return {
                "execution_id": execution_id,
                "status": self.status.value,
                "results": results,
                "execution_time": (self.last_executed - self.created_at).total_seconds()
            }
            
        except Exception as e:
            self.status = FlowStatus.FAILED
            logger.error(f"Flow execution failed: {e}")
            
            execution_record = {
                "execution_id": execution_id,
                "status": self.status.value,
                "error": str(e),
                "start_time": self.created_at.isoformat(),
                "end_time": datetime.now().isoformat()
            }
            self.execution_history.append(execution_record)
            
            return {
                "execution_id": execution_id,
                "status": self.status.value,
                "error": str(e)
            }
    
    async def _execute_from_nodes(self, node_ids: List[str], data_packet: DataPacket) -> Dict[str, Any]:
        """Execute flow starting from specific nodes"""
        results = {}
        
        for node_id in node_ids:
            if node_id in self.nodes:
                try:
                    # Process data through node
                    output_packet = await self.nodes[node_id].process(data_packet)
                    
                    # Store result
                    results[node_id] = {
                        "status": "success" if output_packet.validation_status else "failed",
                        "payload": output_packet.payload,
                        "metadata": output_packet.metadata,
                        "error": output_packet.error_message
                    }
                    
                    # Continue to next nodes if successful
                    if output_packet.validation_status:
                        next_nodes = list(self.graph.successors(node_id))
                        if next_nodes:
                            next_results = await self._execute_from_nodes(next_nodes, output_packet)
                            results.update(next_results)
                    
                except Exception as e:
                    logger.error(f"Error executing node {node_id}: {e}")
                    results[node_id] = {
                        "status": "error",
                        "error": str(e)
                    }
        
        return results
    
    def get_execution_order(self) -> List[str]:
        """Get the topological order of node execution"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            return []
    
    def export_flow_definition(self, format: str = "yaml") -> str:
        """Export flow definition"""
        flow_def = {
            "flow_id": self.flow_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "nodes": {},
            "connections": self.connections,
            "execution_order": self.get_execution_order()
        }
        
        # Add node definitions
        for node_id, node in self.nodes.items():
            flow_def["nodes"][node_id] = {
                "name": node.name,
                "description": node.description,
                "type": node.__class__.__name__,
                "validation_level": node.validation_level.value
            }
        
        if format.lower() == "yaml":
            return yaml.dump(flow_def, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(flow_def, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


class DataFlowManager:
    """Manages multiple data flows and their execution"""
    
    def __init__(self):
        self.flows: Dict[str, DataFlow] = {}
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    def create_flow(self, flow_id: str, name: str, description: str = "") -> DataFlow:
        """Create a new data flow"""
        if flow_id in self.flows:
            raise ValueError(f"Flow with ID {flow_id} already exists")
        
        flow = DataFlow(flow_id, name, description)
        self.flows[flow_id] = flow
        return flow
    
    def get_flow(self, flow_id: str) -> Optional[DataFlow]:
        """Get a flow by ID"""
        return self.flows.get(flow_id)
    
    def list_flows(self) -> List[Dict[str, Any]]:
        """List all flows"""
        return [
            {
                "flow_id": flow.flow_id,
                "name": flow.name,
                "description": flow.description,
                "status": flow.status.value,
                "node_count": len(flow.nodes),
                "created_at": flow.created_at.isoformat(),
                "last_executed": flow.last_executed.isoformat() if flow.last_executed else None
            }
            for flow in self.flows.values()
        ]
    
    async def execute_flow(self, flow_id: str, input_data: Any, start_node_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute a flow"""
        flow = self.get_flow(flow_id)
        if not flow:
            raise ValueError(f"Flow not found: {flow_id}")
        
        # Check if flow is already running
        if flow_id in self.active_executions:
            raise ValueError(f"Flow {flow_id} is already running")
        
        # Execute flow
        task = asyncio.create_task(flow.execute(input_data, start_node_id))
        self.active_executions[flow_id] = task
        
        try:
            result = await task
            return result
        finally:
            # Clean up
            if flow_id in self.active_executions:
                del self.active_executions[flow_id]
    
    def cancel_flow(self, flow_id: str) -> bool:
        """Cancel a running flow"""
        if flow_id in self.active_executions:
            task = self.active_executions[flow_id]
            task.cancel()
            del self.active_executions[flow_id]
            return True
        return False
    
    def export_all_flows(self, format: str = "yaml") -> str:
        """Export all flow definitions"""
        all_flows = {}
        for flow_id, flow in self.flows.items():
            all_flows[flow_id] = {
                "flow_id": flow.flow_id,
                "name": flow.name,
                "description": flow.description,
                "created_at": flow.created_at.isoformat(),
                "nodes": {node_id: {"name": node.name, "type": node.__class__.__name__} 
                         for node_id, node in flow.nodes.items()},
                "connections": flow.connections,
                "execution_order": flow.get_execution_order()
            }
        
        if format.lower() == "yaml":
            return yaml.dump(all_flows, default_flow_style=False)
        elif format.lower() == "json":
            return json.dumps(all_flows, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Utility functions for common data transformations
def create_json_transform(transform_func: Callable[[Dict], Dict]) -> Callable[[Any], Any]:
    """Create a JSON transformation function"""
    def wrapper(data):
        if isinstance(data, dict):
            return transform_func(data)
        return data
    return wrapper


def create_list_filter(filter_func: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Create a list filtering function"""
    def wrapper(data):
        if isinstance(data, list):
            return any(filter_func(item) for item in data)
        return filter_func(data)
    return wrapper


def create_aggregation_sum() -> Callable[[List[Any]], Any]:
    """Create a sum aggregation function"""
    def wrapper(data_list):
        if not data_list:
            return 0
        
        # Try to sum numeric values
        try:
            return sum(item for item in data_list if isinstance(item, (int, float)))
        except TypeError:
            return len(data_list)
    
    return wrapper


def create_aggregation_merge() -> Callable[[List[Any]], Any]:
    """Create a merge aggregation function"""
    def wrapper(data_list):
        if not data_list:
            return []
        
        result = []
        for item in data_list:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        
        return result
    
    return wrapper


# Example usage
async def create_example_flow() -> DataFlow:
    """Create an example data flow"""
    manager = DataFlowManager()
    flow = manager.create_flow("example_flow", "Example Data Processing Flow", 
                              "Demonstrates basic data flow patterns")
    
    # Create nodes
    transform_node = TransformNode(
        "transform_1", 
        "Data Transformer",
        lambda x: {"processed": x, "timestamp": datetime.now().isoformat()}
    )
    
    filter_node = FilterNode(
        "filter_1",
        "Data Filter",
        lambda x: isinstance(x, dict) and "processed" in x
    )
    
    router_node = RouterNode(
        "router_1",
        "Data Router",
        lambda x: "high_priority" if isinstance(x, dict) and x.get("priority") == "high" else "normal"
    )
    router_node.add_route("high_priority", "priority_queue")
    router_node.add_route("normal", "normal_queue")
    
    aggregator_node = AggregatorNode(
        "aggregator_1",
        "Data Aggregator",
        create_aggregation_merge(),
        buffer_size=5
    )
    
    # Add nodes to flow
    flow.add_node(transform_node)
    flow.add_node(filter_node)
    flow.add_node(router_node)
    flow.add_node(aggregator_node)
    
    # Add connections
    flow.add_connection("transform_1", "filter_1")
    flow.add_connection("filter_1", "router_1")
    flow.add_connection("router_1", "aggregator_1")
    
    return flow


if __name__ == "__main__":
    async def main():
        """Example usage of data flows"""
        # Create example flow
        flow = await create_example_flow()
        
        # Execute flow
        input_data = {"message": "Hello World", "priority": "high"}
        result = await flow.execute(input_data)
        
        print("Flow execution result:")
        print(json.dumps(result, indent=2))
        
        # Export flow definition
        flow_def = flow.export_flow_definition("yaml")
        print("\nFlow definition:")
        print(flow_def)
    
    asyncio.run(main())
