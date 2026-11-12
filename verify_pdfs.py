import tarfile
import json

with tarfile.open("datasets/finragbench/pdfs_for_QA/pdf_en.tar.gz", "r:gz") as tar:
    names = tar.getnames()

archive_docs = set()
for n in names:
    filename = n.split("/")[-1]
    if filename.endswith(".pdf"):
        doc_id = filename[:-4]  # remove .pdf
        archive_docs.add(doc_id)

with open("datasets/finragbench/selected_docs.json") as f:
    selected = json.load(f)
selected_docs = selected["selected_docs"]

print("Checking selected docs against archive:\n")
found = []
missing = []
for doc in selected_docs:
    if doc in archive_docs:
        found.append(doc)
        print(f"{doc}")
    else:
        missing.append(doc)
        print(f"{doc}")

print(f"\nFound: {len(found)}/25")
print(f"Missing: {len(missing)}/25")