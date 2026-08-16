import tarfile
import json
import os

os.makedirs("data/raw", exist_ok=True)

# Load selected docs
with open("datasets/finragbench/selected_docs.json") as f:
    selected = json.load(f)
selected_docs = set(selected["selected_docs"])

# Extract only our 25 PDFs
extracted = 0
with tarfile.open("datasets/finragbench/pdfs_for_QA/pdf_en.tar.gz", "r:gz") as tar:
    for member in tar.getmembers():
        filename = member.name.split("/")[-1]
        if filename.endswith(".pdf"):
            doc_id = filename[:-4]
            if doc_id in selected_docs:
                member.name = filename  # flatten path
                tar.extract(member, path="data/raw")
                extracted += 1
                print(f"  Extracted: {filename}")

print(f"\nDone. {extracted}/25 PDFs extracted to data/raw/")