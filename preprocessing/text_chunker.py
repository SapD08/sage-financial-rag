import json
import re
import os
import pdfplumber
from pathlib import Path

CHUNK_SIZE = 400    # words per chunk
CHUNK_OVERLAP = 50  # word overlap between chunks
DATA_RAW = "data/raw"
OUTPUT = "data/chunks/text_chunks.json"

DOCS = [
    "NASDAQ_AMBC_2023", "LSE_HWDN_2022",
    "AIM_CRW_2023", "LSE_HWDN_2020", "NASDAQ_AMBC_2020",
    "NASDAQ_AMBC_2021", "AIM_CRW_2022", "LSE_HWDN_2021",
    "NASDAQ_SRCE_2023", "AIM_CRW_2020", "NASDAQ_SRCE_2021",
    "LSE_HWDN_2023", "AIM_CRW_2021", "NASDAQ_AAON_2021",
    "NASDAQ_SRCE_2022", "NASDAQ_TSBK_2020",
    "FINAL PROOF for Online Posting PDFS_2024ProxyStatement No blank pages",
    "0001471265-24-000009"
]

def clean_text(text):
    text = text.replace("\x00", "").replace("\uf0b7", "-")
    text = re.sub(r"\(cid:\d+\)", "", text)  # remove encoding artifacts
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text(pdf_path, doc_id):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = clean_text(text)
            if len(text.split()) > 20:  # skip near-empty pages
                pages.append({"page": i, "text": text})
    return pages

def chunk_pages(pages, doc_id):
    chunks = []
    chunk_idx = 0
    for page_data in pages:
        words = page_data["text"].split()
        page_num = page_data["page"]
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk_words = words[i: i + CHUNK_SIZE]
            if len(chunk_words) < 20:
                continue
            chunks.append({
                "id":      f"{doc_id}_text_{chunk_idx}",
                "doc_id":  doc_id,
                "page":    page_num,
                "type":    "text",
                "text":    " ".join(chunk_words)
            })
            chunk_idx += 1
    return chunks

def main():
    all_chunks = []
    for doc_id in DOCS:
        pdf_path = f"{DATA_RAW}/{doc_id}.pdf"
        if not os.path.exists(pdf_path):
            print(f"  [SKIP] {doc_id}.pdf not found")
            continue
        print(f"Processing {doc_id}")
        pages = extract_text(pdf_path, doc_id)
        chunks = chunk_pages(pages, doc_id)
        print(f":: {len(pages)} pages, {len(chunks)} chunks")
        all_chunks.extend(chunks)

    os.makedirs("data/chunks", exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()