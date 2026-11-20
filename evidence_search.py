from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode, urlparse

import requests
import trafilatura

from article_extractor import safe_get_public_url

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
HEADERS = {"User-Agent": "TruthCheck-Educational-Prototype/1.0"}


def _article_date_window(value: str) -> tuple[str, str] | None:
    """Return a narrow date window around the submitted article's publication date."""
    if not value or value == "확인되지 않음":
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        try:
            parsed = date.fromisoformat(value[:10])
        except (TypeError, ValueError):
            return None
    return (parsed - timedelta(days=7)).isoformat(), (parsed + timedelta(days=21)).isoformat()


def search_gdelt(query: str, maxrecords: int = 5) -> list[dict]:
    # GDELT DOC ArticleList covers only recent news (not historical events such as 2016).
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


def search_google_news(query: str, maxrecords: int = 5, date_window: tuple[str, str] | None = None) -> list[dict]:
    """Best-effort RSS fallback for archived reporting beyond GDELT DOC's coverage."""
    search_text = query[:140]
    if date_window:
        start, end = date_window
        search_text += f" after:{start} before:{end}"
    params = {
        "q": search_text,
        # The project primarily analyzes Korean corporate news.
        # Search Korean-language coverage in Korea rather than defaulting to US English.
        "hl": "ko",
        "gl": "KR",
        "ceid": "KR:ko",
    }
    response = requests.get(
        f"{GOOGLE_NEWS_RSS}?{urlencode(params)}",
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    results = []
    for item in root.findall("./channel/item")[:maxrecords]:
        title = item.findtext("title") or "제목 없음"
        link = item.findtext("link") or ""
        if not link.startswith(("https://", "http://")):
            continue
        # Resolve the Google News redirect so the report cites the publisher, not the aggregator.
        try:
            resolved = safe_get_public_url(link, timeout=12)
            final_url = resolved.url
            resolved.close()
        except Exception:
            # Keep the RSS result if the publisher redirect cannot be resolved.
            # Its title and snippet remain useful discovery evidence, but the
            # aggregator URL/domain must not be mistaken for the publisher.
            final_url = link
        final_host = (urlparse(final_url).hostname or "").removeprefix("www.")
        publisher_node = item.find("source")
        publisher_name = (publisher_node.text or "").strip() if publisher_node is not None else ""
        url_resolved = bool(final_host and final_host != "news.google.com")
        # If redirect resolution failed, keep the RSS publisher label but
        # do not mistake the Google News host for the publisher.
        domain = final_host if url_resolved else (publisher_name or "원문 매체 URL 미확인")
        description_html = item.findtext("description") or ""
        snippet = BeautifulSoup(description_html, "html.parser").get_text(" ", strip=True)
        published = item.findtext("pubDate") or "날짜 미상"
        try:
            published = parsedate_to_datetime(published).isoformat()
        except (TypeError, ValueError, OverflowError):
            pass
        results.append({
            "title": title,
            "url": final_url,
            "domain": domain,
            "publisher": publisher_name or domain,
            "url_resolved": url_resolved,
            "date": published,
            "language": "ko",
            "country": "KR",
            "snippet": snippet[:1200],
            "search_query": query,
        })
    return results


def fetch_source_text(url: str, max_chars: int = 4500, fallback_text: str = "") -> str:
    try:
        response = safe_get_public_url(url, timeout=12)
        try:
            text = trafilatura.extract(response.text, include_comments=False, include_tables=True, favor_precision=True) or ""
            if text.strip():
                return text.strip()[:max_chars]
        finally:
            response.close()
    except Exception:
        pass
    # Preserve the search engine's short result summary when the publisher
    # blocks extraction or the article is behind a paywall.
    return (fallback_text or "").strip()[:1200]


def collect_evidence(
    search_queries: list[str],
    article_url: str = "",
    max_sources: int = 8,
    progress=None,
    article_date: str = "",
) -> list[dict]:
    all_items: list[dict] = []
    seen_urls = {article_url} if article_url else set()
    # Keep more claim-specific queries so later claims are not silently omitted.
    unique_queries = list(dict.fromkeys(q.strip() for q in search_queries if isinstance(q, str) and q.strip()))[:8]
    date_window = _article_date_window(article_date)

    # GDELT DOC ArtList is limited to recent coverage. Use it for recent articles only.
    try:
        publication_date = date.fromisoformat(article_date[:10]) if article_date else None
    except (TypeError, ValueError):
        publication_date = None
    use_gdelt = publication_date is None or publication_date >= date.today() - timedelta(days=89)

    for idx, query in enumerate(unique_queries):
        if progress:
            progress(f"관련 근거 검색 중 ({idx + 1}/{len(unique_queries)})")
        found: list[dict] = []
        if use_gdelt:
            try:
                found.extend(search_gdelt(query, maxrecords=5))
            except (requests.RequestException, ValueError, ET.ParseError):
                pass
        # RSS fallback also searches the publication period, including older archive material.
        try:
            found.extend(search_google_news(query, maxrecords=5, date_window=date_window))
        except (requests.RequestException, ET.ParseError, ValueError):
            pass
        for item in found:
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                all_items.append(item)
        if idx < len(unique_queries) - 1:
            time.sleep(1.2)

    # Keep multiple results from the same publisher when they are relevant.
    picked = all_items[:max_sources]
    for index, item in enumerate(picked, 1):
        item["id"] = f"S{index}"
        extracted_text = fetch_source_text(item["url"], fallback_text="")
        if extracted_text:
            item["text"] = extracted_text
            item["text_origin"] = "본문"
        elif item.get("snippet"):
            item["text"] = item["snippet"].strip()[:1200]
            item["text_origin"] = "검색 결과 요약"
        else:
            item["text"] = ""
            item["text_origin"] = "본문 확보 실패"
    return picked
