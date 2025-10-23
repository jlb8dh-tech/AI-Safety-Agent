import json, os, pprint

media_path = "../media_agent/media.json"

with open(media_path) as f:
    media = json.load(f)

print("🧩 Integration Test – Media Agent")
print("Found images:")
for img in media["images"]:
    print(f" - {img['file']} | alt: {img['alt']}")

print("\nVideo Summary:")
pprint.pprint(media["video"])
