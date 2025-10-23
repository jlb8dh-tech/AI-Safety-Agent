import json

# Load data from outline + script
with open("../data/outline.json") as f:
    outline = json.load(f)

with open("../data/script.json") as f:
    script = json.load(f)

# Example scene mapping (replace with your real logic)
scenes = [
    {"scene_id": "S1_intro", "goal": "Introduce warehouse environment", "visual_brief": "Forklift backing with beeping alarm"},
    {"scene_id": "S2_hazard", "goal": "Demonstrate blind spot risk", "visual_brief": "Pedestrian walking behind forklift"},
    {"scene_id": "S3_control", "goal": "Show safe reversing with spotter", "visual_brief": "Spotter signaling forklift"}
]

# Save the scene plan
with open("scene_plan.json", "w") as f:
    json.dump({"scenes": scenes}, f, indent=2)

print("✅ Scene plan generated successfully!")
