"""
Outline Pipeline - Structures training topics logically

This module provides functionality to organize training topics into logical hierarchies,
map dependencies between topics, and optimize learning paths.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from pydantic import BaseModel, Field
import yaml
import json
from pathlib import Path


class TopicComplexity(Enum):
    """Enumeration for topic complexity levels"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TopicType(Enum):
    """Enumeration for different types of topics"""
    CONCEPTUAL = "conceptual"
    PRACTICAL = "practical"
    THEORETICAL = "theoretical"
    APPLICATION = "application"
    ASSESSMENT = "assessment"
    INTRODUCTION = "introduction"
    CASE_STUDY = "case_study"
    QUIZ = "quiz"
    DEMONSTRATION = "demonstration"
    HANDS_ON = "hands_on"


@dataclass
class Topic:
    """Represents a single training topic"""
    id: str
    title: str
    description: str
    complexity: TopicComplexity
    topic_type: TopicType
    estimated_duration: int  # in minutes
    prerequisites: Set[str] = field(default_factory=set)
    learning_objectives: List[str] = field(default_factory=list)
    keywords: Set[str] = field(default_factory=set)
    resources: List[str] = field(default_factory=list)
    assessment_criteria: List[str] = field(default_factory=list)


class TopicHierarchy(BaseModel):
    """Represents the hierarchical structure of topics"""
    root_topics: List[str] = Field(default_factory=list)
    topic_graph: Dict[str, List[str]] = Field(default_factory=dict)
    topic_details: Dict[str, Topic] = Field(default_factory=dict)


class OutlinePipeline:
    """
    Pipeline for structuring training topics logically
    
    Features:
    - Hierarchical topic organization
    - Dependency mapping and validation
    - Learning path optimization
    - Topic clustering and grouping
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the outline pipeline"""
        self.config = self._load_config(config_path)
        self.topic_graph = nx.DiGraph()
        self.topic_hierarchy = TopicHierarchy()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {
            "max_prerequisites": 5,
            "complexity_progression": True,
            "duration_threshold": 120,  # minutes
            "clustering_threshold": 0.7
        }
    
    def add_topic(self, topic: Topic) -> bool:
        """
        Add a topic to the pipeline
        
        Args:
            topic: Topic object to add
            
        Returns:
            bool: True if topic was added successfully
        """
        try:
            # Validate topic
            self._validate_topic(topic)
            
            # Add to graph
            self.topic_graph.add_node(topic.id, topic=topic)
            
            # Add prerequisites as edges
            for prereq in topic.prerequisites:
                if prereq in self.topic_graph.nodes:
                    self.topic_graph.add_edge(prereq, topic.id)
                else:
                    # Add placeholder for missing prerequisite
                    self.topic_graph.add_node(prereq, topic=None)
                    self.topic_graph.add_edge(prereq, topic.id)
            
            # Update hierarchy
            self.topic_hierarchy.topic_details[topic.id] = topic
            
            return True
            
        except Exception as e:
            print(f"Error adding topic {topic.id}: {e}")
            return False
    
    def _validate_topic(self, topic: Topic) -> None:
        """Validate topic data"""
        if not topic.id or not topic.title:
            raise ValueError("Topic must have ID and title")
        
        if topic.estimated_duration <= 0:
            raise ValueError("Topic duration must be positive")
        
        if len(topic.prerequisites) > self.config["max_prerequisites"]:
            raise ValueError(f"Too many prerequisites (max: {self.config['max_prerequisites']})")
    
    def build_hierarchy(self) -> TopicHierarchy:
        """
        Build hierarchical structure from topic graph
        
        Returns:
            TopicHierarchy: Structured hierarchy of topics
        """
        # Find root topics (no prerequisites)
        root_topics = [
            node for node in self.topic_graph.nodes()
            if self.topic_graph.in_degree(node) == 0 and self.topic_graph.nodes[node].get('topic')
        ]
        
        # Build topic graph structure
        topic_graph_dict = {}
        for node in self.topic_graph.nodes():
            successors = list(self.topic_graph.successors(node))
            if successors:
                topic_graph_dict[node] = successors
        
        self.topic_hierarchy.root_topics = root_topics
        self.topic_hierarchy.topic_graph = topic_graph_dict
        
        return self.topic_hierarchy
    
    def optimize_learning_path(self, start_topics: List[str], 
                             target_complexity: Optional[TopicComplexity] = None) -> List[str]:
        """
        Optimize learning path based on dependencies and complexity
        
        Args:
            start_topics: List of starting topic IDs
            target_complexity: Optional target complexity level
            
        Returns:
            List[str]: Optimized sequence of topic IDs
        """
        if not start_topics:
            return []
        
        # Use topological sort to respect dependencies
        try:
            # Create subgraph with reachable nodes
            reachable_nodes = set()
            for start in start_topics:
                if start in self.topic_graph:
                    reachable_nodes.update(nx.descendants(self.topic_graph, start))
                    reachable_nodes.add(start)
            
            subgraph = self.topic_graph.subgraph(reachable_nodes)
            topo_order = list(nx.topological_sort(subgraph))
            
            # Filter by complexity if specified
            if target_complexity:
                complexity_order = [TopicComplexity.BEGINNER, TopicComplexity.INTERMEDIATE, 
                                 TopicComplexity.ADVANCED, TopicComplexity.EXPERT]
                target_idx = complexity_order.index(target_complexity)
                
                filtered_order = []
                for topic_id in topo_order:
                    topic = self.topic_graph.nodes[topic_id].get('topic')
                    if topic and complexity_order.index(topic.complexity) <= target_idx:
                        filtered_order.append(topic_id)
                
                return filtered_order
            
            return topo_order
            
        except nx.NetworkXError:
            # Handle cycles in dependency graph
            return self._handle_cycles(start_topics)
    
    def _handle_cycles(self, start_topics: List[str]) -> List[str]:
        """Handle cycles in dependency graph by using DFS"""
        visited = set()
        result = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            
            # Add current node
            if self.topic_graph.nodes[node].get('topic'):
                result.append(node)
            
            # Visit successors
            for successor in self.topic_graph.successors(node):
                dfs(successor)
        
        for start in start_topics:
            if start in self.topic_graph:
                dfs(start)
        
        return result
    
    def cluster_topics(self, similarity_threshold: float = 0.7) -> Dict[str, List[str]]:
        """
        Cluster related topics based on keywords and content similarity
        
        Args:
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            Dict[str, List[str]]: Cluster ID to topic IDs mapping
        """
        clusters = {}
        cluster_id = 0
        
        # Simple keyword-based clustering
        topic_keywords = {}
        for topic_id, topic in self.topic_hierarchy.topic_details.items():
            topic_keywords[topic_id] = topic.keywords
        
        processed = set()
        for topic_id, keywords in topic_keywords.items():
            if topic_id in processed:
                continue
            
            cluster_topics = [topic_id]
            processed.add(topic_id)
            
            # Find similar topics
            for other_id, other_keywords in topic_keywords.items():
                if other_id in processed:
                    continue
                
                similarity = self._calculate_keyword_similarity(keywords, other_keywords)
                if similarity >= similarity_threshold:
                    cluster_topics.append(other_id)
                    processed.add(other_id)
            
            clusters[f"cluster_{cluster_id}"] = cluster_topics
            cluster_id += 1
        
        return clusters
    
    def _calculate_keyword_similarity(self, keywords1: Set[str], keywords2: Set[str]) -> float:
        """Calculate Jaccard similarity between keyword sets"""
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def export_outline(self, format: str = "yaml", output_path: Optional[str] = None) -> str:
        """
        Export the structured outline
        
        Args:
            format: Export format ("yaml", "json")
            output_path: Optional output file path
            
        Returns:
            str: Exported outline content
        """
        # Build hierarchy if not already built
        if not self.topic_hierarchy.root_topics:
            self.build_hierarchy()
        
        # Prepare export data
        export_data = {
            "metadata": {
                "total_topics": len(self.topic_hierarchy.topic_details),
                "root_topics": len(self.topic_hierarchy.root_topics),
                "generated_at": str(Path().cwd())
            },
            "hierarchy": {
                "root_topics": self.topic_hierarchy.root_topics,
                "topic_graph": self.topic_hierarchy.topic_graph
            },
            "topics": {}
        }
        
        # Add topic details
        for topic_id, topic in self.topic_hierarchy.topic_details.items():
            export_data["topics"][topic_id] = {
                "title": topic.title,
                "description": topic.description,
                "complexity": topic.complexity.value,
                "type": topic.topic_type.value,
                "duration": topic.estimated_duration,
                "prerequisites": list(topic.prerequisites),
                "learning_objectives": topic.learning_objectives,
                "keywords": list(topic.keywords),
                "resources": topic.resources,
                "assessment_criteria": topic.assessment_criteria
            }
        
        # Export in requested format
        if format.lower() == "yaml":
            content = yaml.dump(export_data, default_flow_style=False, sort_keys=False)
        elif format.lower() == "json":
            content = json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Write to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
        
        return content
    
    def validate_dependencies(self) -> List[str]:
        """
        Validate topic dependencies and return any issues
        
        Returns:
            List[str]: List of dependency validation issues
        """
        issues = []
        
        # Check for missing prerequisites
        for topic_id, topic in self.topic_hierarchy.topic_details.items():
            for prereq in topic.prerequisites:
                if prereq not in self.topic_hierarchy.topic_details:
                    issues.append(f"Topic '{topic_id}' has missing prerequisite: '{prereq}'")
        
        # Check for circular dependencies
        try:
            nx.topological_sort(self.topic_graph)
        except nx.NetworkXError:
            cycles = list(nx.simple_cycles(self.topic_graph))
            for cycle in cycles:
                issues.append(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        # Check complexity progression
        if self.config.get("complexity_progression", True):
            complexity_order = [TopicComplexity.BEGINNER, TopicComplexity.INTERMEDIATE, 
                             TopicComplexity.ADVANCED, TopicComplexity.EXPERT]
            
            for edge in self.topic_graph.edges():
                source_topic = self.topic_graph.nodes[edge[0]].get('topic')
                target_topic = self.topic_graph.nodes[edge[1]].get('topic')
                
                if source_topic and target_topic:
                    source_idx = complexity_order.index(source_topic.complexity)
                    target_idx = complexity_order.index(target_topic.complexity)
                    
                    if target_idx < source_idx:
                        issues.append(f"Complexity regression: '{edge[0]}' ({source_topic.complexity.value}) -> '{edge[1]}' ({target_topic.complexity.value})")
        
        return issues
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text"""
        import re
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = {word for word in words if word not in stop_words and len(word) > 3}
        
        return keywords
    
    def create_safety_training_outline(self, topic_prompt: str) -> Dict[str, Any]:
        """
        Create a structured safety training outline from a user prompt
        
        Args:
            topic_prompt: User prompt describing the safety topic (e.g., "Forklift operation hazards")
            
        Returns:
            Dict containing structured outline with topics and flow
        """
        # Extract keywords from prompt
        keywords = self._extract_keywords(topic_prompt)
        
        # Determine topic category and create appropriate structure
        if any(word in keywords for word in ["forklift", "lift", "truck", "industrial"]):
            return self._create_forklift_training_outline(topic_prompt, keywords)
        elif any(word in keywords for word in ["hazard", "safety", "risk", "danger"]):
            return self._create_general_safety_outline(topic_prompt, keywords)
        elif any(word in keywords for word in ["equipment", "machine", "tool"]):
            return self._create_equipment_training_outline(topic_prompt, keywords)
        else:
            return self._create_generic_training_outline(topic_prompt, keywords)
    
    def _create_forklift_training_outline(self, prompt: str, keywords: Set[str]) -> Dict[str, Any]:
        """Create forklift-specific training outline"""
        topics = [
            Topic(
                id="forklift_intro",
                title="Introduction to Forklift Operations",
                description="Overview of forklift types, components, and basic operations",
                complexity=TopicComplexity.BEGINNER,
                topic_type=TopicType.INTRODUCTION,
                estimated_duration=15,
                learning_objectives=[
                    "Identify different types of forklifts",
                    "Understand basic forklift components",
                    "Recognize operational requirements"
                ],
                keywords={"forklift", "introduction", "components", "types"}
            ),
            Topic(
                id="forklift_hazards",
                title="Common Forklift Hazards and Risks",
                description="Identification and understanding of common forklift operation hazards",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.THEORETICAL,
                estimated_duration=25,
                prerequisites={"forklift_intro"},
                learning_objectives=[
                    "Identify common forklift hazards",
                    "Understand risk factors",
                    "Recognize warning signs"
                ],
                keywords={"hazards", "risks", "safety", "danger", "warning"}
            ),
            Topic(
                id="forklift_case_study",
                title="Forklift Accident Case Studies",
                description="Real-world case studies of forklift accidents and lessons learned",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.CASE_STUDY,
                estimated_duration=20,
                prerequisites={"forklift_hazards"},
                learning_objectives=[
                    "Analyze real accident scenarios",
                    "Identify contributing factors",
                    "Apply lessons learned"
                ],
                keywords={"case study", "accident", "analysis", "lessons"}
            ),
            Topic(
                id="forklift_safety_procedures",
                title="Forklift Safety Procedures",
                description="Proper safety procedures and best practices for forklift operation",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.PRACTICAL,
                estimated_duration=30,
                prerequisites={"forklift_case_study"},
                learning_objectives=[
                    "Follow safety procedures",
                    "Implement best practices",
                    "Use safety equipment properly"
                ],
                keywords={"procedures", "safety", "best practices", "equipment"}
            ),
            Topic(
                id="forklift_demonstration",
                title="Forklift Operation Demonstration",
                description="Hands-on demonstration of proper forklift operation techniques",
                complexity=TopicComplexity.ADVANCED,
                topic_type=TopicType.DEMONSTRATION,
                estimated_duration=45,
                prerequisites={"forklift_safety_procedures"},
                learning_objectives=[
                    "Demonstrate proper operation",
                    "Practice safety techniques",
                    "Handle various scenarios"
                ],
                keywords={"demonstration", "hands-on", "practice", "operation"}
            ),
            Topic(
                id="forklift_assessment",
                title="Forklift Safety Assessment",
                description="Comprehensive assessment of forklift safety knowledge and skills",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.QUIZ,
                estimated_duration=15,
                prerequisites={"forklift_demonstration"},
                learning_objectives=[
                    "Test knowledge retention",
                    "Assess practical skills",
                    "Identify areas for improvement"
                ],
                keywords={"assessment", "quiz", "test", "evaluation"}
            )
        ]
        
        # Add topics to pipeline
        for topic in topics:
            self.add_topic(topic)
        
        # Build hierarchy
        hierarchy = self.build_hierarchy()
        
        # Create learning flow
        learning_flow = [
            "forklift_intro",
            "forklift_hazards", 
            "forklift_case_study",
            "forklift_safety_procedures",
            "forklift_demonstration",
            "forklift_assessment"
        ]
        
        return {
            "prompt": prompt,
            "topic_category": "forklift_safety",
            "learning_flow": learning_flow,
            "total_duration": sum(topic.estimated_duration for topic in topics),
            "topics": [
                {
                    "id": topic.id,
                    "title": topic.title,
                    "type": topic.topic_type.value,
                    "duration": topic.estimated_duration,
                    "description": topic.description
                }
                for topic in topics
            ],
            "flow_summary": "Intro → Common Risks → Case Study → Safety Procedures → Demonstration → Quiz"
        }
    
    def _create_general_safety_outline(self, prompt: str, keywords: Set[str]) -> Dict[str, Any]:
        """Create general safety training outline"""
        topics = [
            Topic(
                id="safety_intro",
                title="Safety Introduction",
                description="Introduction to workplace safety concepts and importance",
                complexity=TopicComplexity.BEGINNER,
                topic_type=TopicType.INTRODUCTION,
                estimated_duration=10,
                learning_objectives=["Understand safety importance", "Learn basic safety concepts"],
                keywords={"safety", "introduction", "workplace"}
            ),
            Topic(
                id="hazard_identification",
                title="Hazard Identification",
                description="How to identify and assess workplace hazards",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.THEORETICAL,
                estimated_duration=20,
                prerequisites={"safety_intro"},
                learning_objectives=["Identify hazards", "Assess risks"],
                keywords={"hazard", "identification", "risk", "assessment"}
            ),
            Topic(
                id="safety_case_study",
                title="Safety Case Studies",
                description="Real-world safety incidents and lessons learned",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.CASE_STUDY,
                estimated_duration=15,
                prerequisites={"hazard_identification"},
                learning_objectives=["Analyze incidents", "Learn from mistakes"],
                keywords={"case study", "incident", "analysis"}
            ),
            Topic(
                id="safety_quiz",
                title="Safety Knowledge Assessment",
                description="Assessment of safety knowledge and understanding",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.QUIZ,
                estimated_duration=10,
                prerequisites={"safety_case_study"},
                learning_objectives=["Test knowledge", "Identify gaps"],
                keywords={"quiz", "assessment", "knowledge"}
            )
        ]
        
        for topic in topics:
            self.add_topic(topic)
        
        hierarchy = self.build_hierarchy()
        
        return {
            "prompt": prompt,
            "topic_category": "general_safety",
            "learning_flow": ["safety_intro", "hazard_identification", "safety_case_study", "safety_quiz"],
            "total_duration": sum(topic.estimated_duration for topic in topics),
            "topics": [
                {
                    "id": topic.id,
                    "title": topic.title,
                    "type": topic.topic_type.value,
                    "duration": topic.estimated_duration,
                    "description": topic.description
                }
                for topic in topics
            ],
            "flow_summary": "Intro → Hazard ID → Case Study → Quiz"
        }
    
    def _create_equipment_training_outline(self, prompt: str, keywords: Set[str]) -> Dict[str, Any]:
        """Create equipment-specific training outline"""
        topics = [
            Topic(
                id="equipment_intro",
                title="Equipment Introduction",
                description="Introduction to equipment types and basic operation",
                complexity=TopicComplexity.BEGINNER,
                topic_type=TopicType.INTRODUCTION,
                estimated_duration=15,
                learning_objectives=["Identify equipment", "Understand basics"],
                keywords={"equipment", "introduction", "operation"}
            ),
            Topic(
                id="equipment_safety",
                title="Equipment Safety Procedures",
                description="Safety procedures and precautions for equipment operation",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.PRACTICAL,
                estimated_duration=25,
                prerequisites={"equipment_intro"},
                learning_objectives=["Follow safety procedures", "Use safety equipment"],
                keywords={"safety", "procedures", "precautions"}
            ),
            Topic(
                id="equipment_demo",
                title="Equipment Demonstration",
                description="Hands-on demonstration of proper equipment operation",
                complexity=TopicComplexity.ADVANCED,
                topic_type=TopicType.DEMONSTRATION,
                estimated_duration=30,
                prerequisites={"equipment_safety"},
                learning_objectives=["Demonstrate operation", "Practice techniques"],
                keywords={"demonstration", "hands-on", "practice"}
            ),
            Topic(
                id="equipment_assessment",
                title="Equipment Operation Assessment",
                description="Assessment of equipment operation skills and knowledge",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.QUIZ,
                estimated_duration=15,
                prerequisites={"equipment_demo"},
                learning_objectives=["Test skills", "Assess knowledge"],
                keywords={"assessment", "quiz", "skills"}
            )
        ]
        
        for topic in topics:
            self.add_topic(topic)
        
        return {
            "prompt": prompt,
            "topic_category": "equipment_training",
            "learning_flow": ["equipment_intro", "equipment_safety", "equipment_demo", "equipment_assessment"],
            "total_duration": sum(topic.estimated_duration for topic in topics),
            "topics": [
                {
                    "id": topic.id,
                    "title": topic.title,
                    "type": topic.topic_type.value,
                    "duration": topic.estimated_duration,
                    "description": topic.description
                }
                for topic in topics
            ],
            "flow_summary": "Intro → Safety → Demo → Assessment"
        }
    
    def _create_generic_training_outline(self, prompt: str, keywords: Set[str]) -> Dict[str, Any]:
        """Create generic training outline for unknown topics"""
        topics = [
            Topic(
                id="topic_intro",
                title=f"Introduction to {prompt.title()}",
                description=f"Introduction to {prompt}",
                complexity=TopicComplexity.BEGINNER,
                topic_type=TopicType.INTRODUCTION,
                estimated_duration=15,
                learning_objectives=[f"Understand {prompt}", "Learn basics"],
                keywords=keywords
            ),
            Topic(
                id="topic_content",
                title=f"{prompt.title()} Content",
                description=f"Main content about {prompt}",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.THEORETICAL,
                estimated_duration=25,
                prerequisites={"topic_intro"},
                learning_objectives=[f"Master {prompt}", "Apply knowledge"],
                keywords=keywords
            ),
            Topic(
                id="topic_assessment",
                title=f"{prompt.title()} Assessment",
                description=f"Assessment of {prompt} knowledge",
                complexity=TopicComplexity.INTERMEDIATE,
                topic_type=TopicType.QUIZ,
                estimated_duration=10,
                prerequisites={"topic_content"},
                learning_objectives=["Test knowledge", "Evaluate understanding"],
                keywords=keywords
            )
        ]
        
        for topic in topics:
            self.add_topic(topic)
        
        return {
            "prompt": prompt,
            "topic_category": "generic_training",
            "learning_flow": ["topic_intro", "topic_content", "topic_assessment"],
            "total_duration": sum(topic.estimated_duration for topic in topics),
            "topics": [
                {
                    "id": topic.id,
                    "title": topic.title,
                    "type": topic.topic_type.value,
                    "duration": topic.estimated_duration,
                    "description": topic.description
                }
                for topic in topics
            ],
            "flow_summary": "Intro → Content → Assessment"
        }


# Example usage and factory functions
def create_sample_topics() -> List[Topic]:
    """Create sample topics for testing"""
    topics = [
        Topic(
            id="python_basics",
            title="Python Basics",
            description="Introduction to Python programming fundamentals",
            complexity=TopicComplexity.BEGINNER,
            topic_type=TopicType.CONCEPTUAL,
            estimated_duration=90,
            learning_objectives=["Understand Python syntax", "Write basic programs"],
            keywords={"python", "programming", "basics", "syntax"}
        ),
        Topic(
            id="data_structures",
            title="Data Structures",
            description="Understanding lists, dictionaries, and other data structures",
            complexity=TopicComplexity.INTERMEDIATE,
            topic_type=TopicType.THEORETICAL,
            estimated_duration=120,
            prerequisites={"python_basics"},
            learning_objectives=["Master data structures", "Choose appropriate structures"],
            keywords={"data structures", "lists", "dictionaries", "algorithms"}
        ),
        Topic(
            id="oop_concepts",
            title="Object-Oriented Programming",
            description="Classes, objects, inheritance, and polymorphism",
            complexity=TopicComplexity.INTERMEDIATE,
            topic_type=TopicType.CONCEPTUAL,
            estimated_duration=150,
            prerequisites={"python_basics"},
            learning_objectives=["Design classes", "Implement inheritance"],
            keywords={"oop", "classes", "objects", "inheritance"}
        ),
        Topic(
            id="advanced_python",
            title="Advanced Python",
            description="Advanced Python features and best practices",
            complexity=TopicComplexity.ADVANCED,
            topic_type=TopicType.APPLICATION,
            estimated_duration=180,
            prerequisites={"data_structures", "oop_concepts"},
            learning_objectives=["Use advanced features", "Apply best practices"],
            keywords={"advanced", "decorators", "generators", "best practices"}
        )
    ]
    
    return topics


if __name__ == "__main__":
    # Example usage
    pipeline = OutlinePipeline()
    
    # Add sample topics
    for topic in create_sample_topics():
        pipeline.add_topic(topic)
    
    # Build hierarchy
    hierarchy = pipeline.build_hierarchy()
    
    # Optimize learning path
    learning_path = pipeline.optimize_learning_path(["python_basics"])
    print("Learning path:", learning_path)
    
    # Validate dependencies
    issues = pipeline.validate_dependencies()
    if issues:
        print("Dependency issues:", issues)
    
    # Export outline
    outline = pipeline.export_outline("yaml")
    print("Generated outline:")
    print(outline)
