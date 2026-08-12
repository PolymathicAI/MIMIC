"""Download swissprot"""
import argparse
import requests
import re
import json
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,id,protein_name,gene_names,organism_id,length,cc_function,xref_interpro"
PAGE_SIZE = 500
HEADERS = {"Accept": "application/json"}
re_next_link = re.compile(r'<(.+)>; rel="next"')


def build_query(organisms: Optional[List[str]] = None) -> str:
    parts = ["reviewed:true"]
    if organisms:
        orgs = " OR ".join(f'organism_id:\"{o}\"' for o in organisms)
        parts.append(f"({orgs})")
    return " AND ".join(parts)


def get_next_link(headers) -> Optional[str]:
    link = headers.get("Link", "")
    match = re_next_link.search(link)
    return match.group(1) if match else None


def fetch_page(url: str) -> List[dict]:
    if "format=" not in url:
        url += "&format=json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()["results"]


def collect_all_page_urls(initial_url: str) -> List[str]:
    urls = []
    next_url = initial_url

    while next_url:
        resp = requests.get(next_url, headers=HEADERS)
        resp.raise_for_status()
        urls.append(next_url)
        next_url = get_next_link(resp.headers)

    return urls


def download_all(query: str, num_proc: int = 8) -> List[dict]:
    base_url = f"{BASE_URL}?query={query}&fields={FIELDS}&format=json&size={PAGE_SIZE}"
    print("Collecting all paginated URLs...")
    all_urls = collect_all_page_urls(base_url)

    print(f"Found {len(all_urls)} pages. Downloading in parallel...")

    results = []
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        futures = {executor.submit(fetch_page, url): url for url in all_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading pages"):
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Error downloading a page: {e}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Download SwissProt records from UniProt")
    parser.add_argument("--organisms", "-o", nargs="*", help="List of organism IDs (e.g. 9606 10090)")
    parser.add_argument("--output", "-f", required=True, help="Output filename (JSON)")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of parallel processes (default: 8)")
    args = parser.parse_args()

    query = build_query(args.organisms if args.organisms else None)
    print(f"Running query: {query}")

    data = download_all(query, num_proc=args.num_proc)

    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} records to {args.output}")


if __name__ == "__main__":
    main()