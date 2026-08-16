import json
from collections import Counter

with open("eval/results.json") as f:
    results = json.load(f)

print(f"Total results: {len(results)}")

by_config = Counter(r["config"] for r in results)
print("\nBy config:")
for config, count in by_config.items():
    print(f"  {config}: {count}")

# Check for duplicates
keys = [(r["query_id"], r["config"]) for r in results]
duplicates = len(keys) - len(set(keys))
print(f"\nDuplicates: {duplicates}")