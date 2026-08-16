import json

with open("eval/results.json") as f:
    results = json.load(f)

filtered = [r for r in results if r["config"] != "sage"]
print(f"Removed {len(results) - len(filtered)} SAGE results")
print(f"Keeping {len(filtered)} results")

with open("eval/results.json", "w") as f:
    json.dump(filtered, f, indent=2)

print("Done")