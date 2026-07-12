#!/usr/bin/env python3
"""Download a varied set of 100 ML-to-LLM research papers from arXiv."""

import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

NUMBER_OF_PAPERS = 100
OUTPUT_FOLDER = Path("papers")
API = "https://export.arxiv.org/api/query"
USER_AGENT = "Personal research paper downloader/1.0"
NAMESPACE = {"a": "http://www.w3.org/2005/Atom"}

# Results are taken evenly from these areas. "relevance" is used instead of
# publication date, so the collection includes older/basic and newer papers.
TOPICS = [
    '(all:"supervised learning" OR all:"machine learning fundamentals")',
    '(all:"linear regression" OR all:"logistic regression")',
    '(all:"decision tree" OR all:"random forest" OR all:"gradient boosting")',
    '(all:"support vector machine" OR all:"kernel method")',
    '(all:clustering OR all:"k-means" OR all:"dimensionality reduction")',
    '(all:"bayesian learning" OR all:"probabilistic graphical model")',
    '(all:"neural network" OR all:"deep learning" OR all:backpropagation)',
    '(all:"convolutional neural network" OR all:"computer vision")',
    '(all:"recurrent neural network" OR all:LSTM OR all:"sequence model")',
    '(all:"reinforcement learning" OR all:"Q-learning")',
    '(all:transformer OR all:"attention mechanism" OR all:BERT)',
    '(all:"large language model" OR all:"instruction tuning" OR all:RAG)',
]


def safe_filename(title, paper_id):
    """Use the official arXiv title while removing invalid filename symbols."""
    title = " ".join(title.split())
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip(" .")
    suffix = f" [arXiv {paper_id}].pdf"
    return title[: 190 - len(suffix)].rstrip(" .") + suffix


def get_papers(query):
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": 25,
        "sortBy": "relevance",       # Not newest-only
        "sortOrder": "descending",
    })
    request = urllib.request.Request(
        f"{API}?{params}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        root = ET.fromstring(response.read())

    papers = []
    for entry in root.findall("a:entry", NAMESPACE):
        title = entry.findtext("a:title", default="Untitled", namespaces=NAMESPACE)
        paper_id = entry.findtext("a:id", default="", namespaces=NAMESPACE)
        paper_id = paper_id.rstrip("/").split("/")[-1]
        base_id = re.sub(r"v\d+$", "", paper_id)
        papers.append((title, paper_id, base_id))
    return papers


def download(title, paper_id):
    path = OUTPUT_FOLDER / safe_filename(title, paper_id)
    if path.exists() and path.stat().st_size > 1000:
        return path

    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("arXiv did not return a PDF")

    path.write_bytes(data)
    return path


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Collect relevant candidates for every topic.
    topic_results = []
    for number, query in enumerate(TOPICS, 1):
        print(f"Finding papers for topic {number}/{len(TOPICS)}...")
        try:
            topic_results.append(get_papers(query))
        except Exception as error:
            print(f"  Topic skipped: {error}")
            topic_results.append([])
        if number < len(TOPICS):
            time.sleep(3)  # Be polite to the arXiv API.

    # Round-robin selection ensures all topics are represented rather than
    # allowing deep learning or LLM results to dominate the collection.
    candidates = []
    seen = set()
    for position in range(25):
        for results in topic_results:
            if position < len(results):
                paper = results[position]
                if paper[2] not in seen:
                    seen.add(paper[2])
                    candidates.append(paper)

    downloaded = 0
    for title, paper_id, _ in candidates:
        if downloaded >= NUMBER_OF_PAPERS:
            break
        try:
            path = download(title, paper_id)
            downloaded += 1
            print(f"[{downloaded}/{NUMBER_OF_PAPERS}] {path.name}")
            time.sleep(1)
        except Exception as error:
            print(f"Skipped {paper_id}: {error}")

    print(f"\nDone. {downloaded} PDFs are in: {OUTPUT_FOLDER.resolve()}")


if __name__ == "__main__":
    main()
