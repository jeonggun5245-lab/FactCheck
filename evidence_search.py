from __future__ import annotations

import time
from urllib.parse import urlparse

import requests
import trafilatura

from article_extractor import safe_get_public_url

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
HEADERS = {"User-Agent": "TruthCheck-Educational-Prototype/1.0"}


def search_gdelt(query: str, maxrecords: int = 5) -> list[dict]:
    params = {
        "query": query[:180],
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "sort": "HybridRel",
        "timespan": "3m",
    }
    response = requests.get(GDELT_URL, params=params, headers=HEADERS, timeout=25)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    results = []
    for item in articles:
        url = item.get("url", "")
        if not url.startswith(("https://", "http://")):
            continue
        results.append({
            "title": item.get("title") or "제목 없음",
            "url": url,
            "domain": item.get("domain") or (urlparse(url).hostname or "출처 미상"),
            "date": item.get("seendate") or "날짜 미상",
            "language": item.get("language") or "",
            "country": item.get("sourcecountry") or "",
        })
    return results


def fetch_source_text(url: str, max_chars: int = 4500) -> str:
    try:
        response = safe_get_public_url(url, timeout=12)
        text = trafilatura.extract(response.text, include_comments=False, include_tables=True, favor_precision=True) or ""
        return text.strip()[:max_chars]
    except Exception:
        return ""


def collect_evidence(search_queries: list[str], article_url: str = "", max_sources: int = 8, progress=None) -> list[dict]:
    all_items: list[dict] = []
    seen_urls = {article_url} if article_url else set()
    unique_queries = list(dict.fromkeys(q.strip() for q in search_queries if isinstance(q, str) and q.strip()))[:3]

    for idx, query in enumerate(unique_queries):
        if progress:
            progress(f"관련 근거 검색 중 ({idx + 1}/{len(unique_queries)})")
        try:
            found = search_gdelt(query, maxrecords=5)
        except requests.RequestException:
            found = []
        for item in found:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
        if idx < len(unique_queries) - 1:
            time.sleep(5.2)

    picked: list[dict] = []
    domains: set[str] = set()
    for item in all_items:
        domain = item.get("domain", "")
        if domain and domain not in domains:
            domains.add(domain)
            picked.append(item)
        if len(picked) >= max_sources:
            break

    for index, item in enumerate(picked, 1):
        item["id"] = f"S{index}"
        item["text"] = fetch_source_text(item["url"])
    return picked
