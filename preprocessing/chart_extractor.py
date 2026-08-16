import json
import os
import base64
import io
import sys
import pdfplumber

sys.path.insert(0, ".")
from config import OPENAI_API_KEY
from openai import OpenAI

DATA_RAW = "data/raw"
OUTPUT = "data/chunks/figure_chunks.json"
CACHE_DIR = "data/figures"

os.makedirs(CACHE_DIR, exist_ok=True)

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

CHART_PROMPT = """This is a page from a financial document.
Identify and describe ALL charts, graphs or figures on this page.

For each chart:
1. Write ## [Chart Title]
2. Type: Bar chart / Line chart / Pie chart / etc.
3. Extract ALL data points with exact values and labels
4. Describe the key trend in 1-2 sentences
5. Include axis labels and units

If there are NO charts or figures on this page, respond only with: NO_CHART"""

client = OpenAI(api_key=OPENAI_API_KEY)

def has_chart(page):
    text = page.extract_text() or ""
    word_count = len(text.split())
    images = page.images
    large_images = [img for img in images
                    if img["width"] > 150 and img["height"] > 100]
    if word_count < 150 and len(large_images) >= 1:
        return True
    chart_keywords = ["figure", "chart", "graph", "exhibit", "trend"]
    if any(kw in text.lower() for kw in chart_keywords):
        return len(large_images) >= 1
    return False

def page_to_base64(page):
    img = page.to_image(resolution=150)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def extract_chart_from_page(doc_id, page_num, page):
    cache_file = f"{CACHE_DIR}/{doc_id}_p{page_num}.md"
    if os.path.exists(cache_file):
        return open(cache_file).read()

    img_b64 = page_to_base64(page)
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CHART_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "high"
                }}
            ]
        }]
    )
    result = response.choices[0].message.content
    open(cache_file, "w").write(result)
    return result

def process_doc(doc_id):
    pdf_path = f"{DATA_RAW}/{doc_id}.pdf"
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            if not has_chart(page):
                continue
            print(f"  Page {i}: chart detected")
            result = extract_chart_from_page(doc_id, i, page)
            if result.strip().upper() == "NO_CHART":
                continue
            chunks.append({
                "id":     f"{doc_id}_fig_p{i}",
                "doc_id": doc_id,
                "page":   i,
                "type":   "figure",
                "text":   result
            })
    return chunks

def main():
    all_chunks = []
    for doc_id in DOCS:
        print(f"\nProcessing {doc_id}")
        chunks = process_doc(doc_id)
        print(f"  → {len(chunks)} figure chunks")
        all_chunks.extend(chunks)

    with open(OUTPUT, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal figure chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    main()