import json

total = 0
for chunk_type in ["text", "table", "figure"]:
    with open(f"data/chunks/{chunk_type}_chunks.json") as f:
        chunks = json.load(f)
    print(f"{chunk_type}_chunks: {len(chunks)}")
    total += len(chunks)

print(f"\nTotal chunks: {total}")