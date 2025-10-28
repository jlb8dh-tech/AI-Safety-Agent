# Black Unicorn Agent - Safety Training Pipelines

A modular agent system for workplace safety training with intelligent outline generation and regulatory standards lookup.

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs data exports

# Test the system
python test_enhanced.py
```

### Basic Usage

**Generate Forklift Training Outline:**
```bash
python main.py --mode safety --format json
```

Input: `"Forklift operation hazards"`  
Output: `"Intro → Common Risks → Case Study → Safety Procedures → Demonstration → Quiz"`

**Standards Lookup:**
```bash
python main.py --mode safety
```

Input: `"Forklift operation"`  
Output: `"1910.178 – Powered Industrial Trucks" (OSHA mandatory)`

## Components

### 1. Outline Pipeline (`agent/pipelines/outline/`)
Generates structured learning outlines from user prompts.

**Usage:**
```python
from agent.pipelines.outline import OutlinePipeline

pipeline = OutlinePipeline()
outline = pipeline.create_safety_training_outline("Forklift operation hazards")

print(outline['flow_summary'])  
# "Intro → Common Risks → Case Study → Safety Procedures → Demonstration → Quiz"
```

**Features:**
- Automatic topic detection (forklift, electrical, equipment, fire safety)
- 6-step training curriculum generation
- Duration estimation (150 minutes for forklift)
- Dependency mapping between topics

### 2. Standards Pipeline (`agent/pipelines/standards/`)
Retrieves OSHA, NIOSH, NFPA, and ANSI standards.

**Usage:**
```python
from agent.pipelines.standards import StandardsLookupPipeline

pipeline = StandardsLookupPipeline()
standards = pipeline.lookup_workplace_safety_standards("Forklift operation")

for std in standards:
    print(f"{std.standard_number} – {std.title}")
    print(f"Organization: {std.organization.value}")
    print(f"Compliance: {std.compliance_level.value}")
```

**Supported Organizations:**
- OSHA (Occupational Safety and Health Administration)
- NIOSH (National Institute for Occupational Safety and Health)
- NFPA (National Fire Protection Association)
- ANSI (American National Standards Institute)

### 3. Data Flows (`agent/pipelines/data_flows/`)
Orchestrates data processing through clean, documented workflows.

**Usage:**
```python
from agent.pipelines.data_flows import DataFlowManager, TransformNode

manager = DataFlowManager()
flow = manager.create_flow("my_flow", "My Flow")

# Add processing nodes
node = TransformNode("process", "Process Data", lambda x: {"processed": x})
flow.add_node(node)

# Execute flow
result = await flow.execute({"input": "data"})
```

## Integration for Team Members

### Python Integration

**Import the agent:**
```python
from agent.pipelines.outline import OutlinePipeline
from agent.pipelines.standards import StandardsLookupPipeline
from agent.pipelines.data_flows import DataFlowManager
```

**Create training outlines:**
```python
# Initialize outline pipeline
outline_pipeline = OutlinePipeline()

# Generate safety training outline
result = outline_pipeline.create_safety_training_outline(
    "Electrical safety hazards"
)

# Access the learning flow
flow = result['flow_summary']  # "Intro → Hazard ID → Case Study → Quiz"
topics = result['topics']      # List of topic details
duration = result['total_duration']  # Total minutes
```

**Look up safety standards:**
```python
# Initialize standards pipeline
standards_pipeline = StandardsLookupPipeline()

# Look up relevant standards
standards = standards_pipeline.lookup_workplace_safety_standards(
    "Forklift operation"
)

# Access standard details
for standard in standards:
    print(f"{standard.standard_number}: {standard.title}")
    print(f"Requirements: {standard.requirements}")
    print(f"Penalties: {standard.penalties}")
```

**Orchestrate workflows:**
```python
# Create data flow manager
flow_manager = DataFlowManager()

# Create custom flow
my_flow = flow_manager.create_flow(
    "training_flow", 
    "Training Integration Flow"
)

# Add nodes
transform = TransformNode(
    "process_topics",
    "Process Topics",
    lambda data: {"processed": data.get("topics", [])}
)
my_flow.add_node(transform)

# Execute
result = await flow_manager.execute_flow("training_flow", your_data)
```

### API Integration (Coming Soon)
```python
# Future REST API endpoint
POST /api/v1/safety-training
{
  "prompt": "Forklift operation hazards",
  "include_standards": true
}

# Response
{
  "outline": {
    "flow_summary": "Intro → Common Risks → Case Study → Quiz",
    "duration": 150,
    "topics": [...]
  },
  "standards": [
    {"number": "1910.178", "title": "Powered Industrial Trucks", ...}
  ]
}
```

### Configuration

Edit `agent/config/config.yaml` to customize:
- Update intervals for standards
- Learning path preferences
- Validation levels
- API endpoints

## Command-Line Interface

```bash
# Safety training mode
python main.py --mode safety --format json

# Outline generation only
python main.py --mode outline --input topics.json --output outline.yaml

# Compliance checking
python main.py --mode compliance --input org_profile.json

# Integrated workflow
python main.py --mode integrated --input data.json --output results.yaml
```

## Examples

### Comprehensive Forklift Demo
```bash
python forklift_example.py
```

This demonstrates:
- Training outline generation
- Standards lookup integration
- Data flow orchestration
- Complete workflow with export

### Run All Examples
```bash
python examples.py
```

Includes examples for:
- Outline pipeline
- Standards lookup
- Data flows
- Integration workflows

## Testing

```bash
# Run all tests
python test_enhanced.py

# Run unit tests
python -m pytest tests/ -v

# Test specific component
python -m pytest tests/test_agent.py::TestOutlinePipeline
```

## Project Structure

```
agent/
├── pipelines/
│   ├── outline/          # Training topic structuring
│   ├── standards/        # OSHA/NIOSH/NFPA/ANSI lookup
│   └── data_flows/      # Workflow orchestration
├── config/              # Configuration files
├── utils/               # Helper functions
└── tests/               # Test suites

main.py                  # CLI entry point
forklift_example.py      # Comprehensive demo
examples.py              # Usage examples
test_enhanced.py         # Component tests
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- networkx (graph processing)
- pydantic (data validation)
- yaml (configuration)
- pytest (testing)

## Support

For questions or integration help:
1. Review component documentation in `agent/pipelines/`
2. Check examples in `examples.py` and `forklift_example.py`
3. Run tests: `python test_enhanced.py`
4. See inline code documentation for API details

## License

MIT License - See LICENSE file for details