from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup

USER_AGENT = "Mozilla/5.0 (compatible; TruthCheckDemo/1.0)"


def validate_public_http_url(value: str) -> tuple[bool, str]:
    try:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return False, "http:// 또는 https://로 시작하는 기사 주소를 입력해 주세요."
        host = parsed.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
            return False, "공개 인터넷 주소만 가져올 수 있습니다."
        infos = socket.getaddrinfo(host, None)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                return False, "공개 인터넷 주소만 가져올 수 있습니다."
        return True, ""
    except socket.gaierror:
        return False, "주소의 서버를 찾을 수 없습니다."
    except Exception:
        return False, "기사 주소를 확인할 수 없습니다."


def safe_get_public_url(url: str, timeout: int = 20) -> requests.Response:
    """GET a public HTTP(S) URL while validating every redirect target."""
    current = url.strip()
    for _ in range(6):
        ok, reason = validate_public_http_url(current)
        if not ok:
            raise ValueError(reason)
        response = requests.get(
            current, headers={"User-Agent": USER_AGENT}, timeout=timeout,
            allow_redirects=False,
        )
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("기사 주소의 이동 위치를 확인할 수 없습니다.")
            current = urljoin(current, location)
            continue
        response.raise_for_status()
        return response
    raise ValueError("기사 주소의 이동 횟수가 너무 많습니다.")


def extract_article(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("기사 URL 또는 본문을 입력해 주세요.")

    if raw.lower().startswith(("http://", "https://")):
        ok, reason = validate_public_http_url(raw)
        if not ok:
            raise ValueError(reason)
        response = requests.get(raw, headers={"User-Agent": USER_AGENT}, timeout=20, allow_redirects=True)
        response.raise_for_status()
        final_url = response.url
        ok, reason = validate_public_http_url(final_url)
        if not ok:
            raise ValueError(reason)

        html = response.text
        metadata = trafilatura.extract_metadata(html)
        body = trafilatura.extract(html, include_comments=False, include_tables=True, favor_precision=True) or ""
        soup = BeautifulSoup(html, "html.parser")
        title = metadata.title if metadata and metadata.title else ""
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)
        if len(body.strip()) < 120:
            raise ValueError("기사 본문을 충분히 가져오지 못했습니다. 유료벽·로그인·사이트 차단일 수 있으니 기사 본문을 직접 붙여 넣어 주세요.")
        date = getattr(metadata, "date", None) if metadata else None
        author = getattr(metadata, "author", None) if metadata else None
        return {"title": title or "제목 미확인", "url": final_url, "body": body.strip()[:24000], "date": date or "확인되지 않음", "author": author or "확인되지 않음"}

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return {"title": lines[0][:200] if lines else "제목 미입력", "url": "", "body": raw[:24000], "date": "확인되지 않음", "author": "확인되지 않음"}
