from __future__ import annotations

import json
import time
from typing import Any

from google import genai
from google.genai import types

from scoring import SCORE_KEYS, VALIDITY_KEYS, clamp_score_map, score_index, clean_source_ids

DEFAULT_MODEL = "gemini-3.8-flash"

CLAIM_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "claims": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "claim": {"type": "STRING"},
            "type": {"type": "STRING", "enum": ["사실 주장", "의견·평가", "전망·예측", "해석"]},
            "search_query": {"type": "STRING"},
        }, "required": ["claim", "type", "search_query"]}},
        "companies": {"type": "ARRAY", "items": {"type": "STRING"}},
        "article_summary": {"type": "STRING"},
    },
    "required": ["claims", "companies", "article_summary"],
}

FINAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {"type": "STRING"},
        "claims": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "claim": {"type": "STRING"},
            "type": {"type": "STRING"},
            "verdict": {"type": "STRING", "enum": ["확인", "일부 확인·맥락 누락", "근거 부족", "허위·반박됨", "평가 보류", "의견·전망"]},
            "reason": {"type": "STRING"},
            "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
        }, "required": ["claim", "type", "verdict", "reason", "source_ids"]}},
        "reliability": {"type": "OBJECT", "properties": {key: {"type": ["NUMBER", "NULL"], "enum": [0, 0.5, 1, None]} for key in SCORE_KEYS}, "required": SCORE_KEYS},
        "validity": {"type": "OBJECT", "properties": {key: {"type": ["NUMBER", "NULL"], "enum": [0, 0.5, 1, None]} for key in VALIDITY_KEYS}, "required": VALIDITY_KEYS},
        "reliability_notes": {"type": "OBJECT", "properties": {key: {"type": "STRING"} for key in SCORE_KEYS}, "required": SCORE_KEYS},
        "validity_notes": {"type": "OBJECT", "properties": {key: {"type": "STRING"} for key in VALIDITY_KEYS}, "required": VALIDITY_KEYS},
        "tone": {"type": "STRING", "enum": ["긍정", "부정", "중립", "복합", "판단 보류"]},
        "bias_signals": {"type": "ARRAY", "items": {"type": "STRING"}},
        "bias_explanation": {"type": "STRING"},
        "companies": {"type": "ARRAY", "items": {"type": "STRING"}},
        "audience_impact": {"type": "STRING"},
        "corporate_impact": {"type": "STRING"},
        "observed_market_change": {"type": "STRING"},
        "documented_financial_impact": {"type": "STRING"},
        "causality_assessment": {"type": "STRING"},
        "limitations": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "claims", "reliability", "validity", "reliability_notes", "validity_notes", "tone", "bias_signals", "bias_explanation", "companies", "audience_impact", "corporate_impact", "observed_market_change", "documented_financial_impact", "causality_assessment", "limitations"],
}


def _generate_json(client: genai.Client, model: str, prompt: str, schema: dict, max_output_tokens: int = 7000) -> dict[str, Any]:
    # Retry transient server/deadline failures with bounded backoff.
    # Authentication, quota, and malformed-request errors surface immediately.
    response = None
    retry_delays = (5, 15)
    for attempt in range(len(retry_delays) + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    temperature=0.1,
                    max_output_tokens=max_output_tokens,
                ),
            )
            break
        except Exception as exc:
            status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
            message = str(exc)
            is_transient = (
                status_code in (408, 429, 500, 502, 503, 504)
                or any(code in message for code in (
                    "408", "429", "500", "502", "503", "504",
                    "DEADLINE_EXCEEDED", "UNAVAILABLE",
                ))
                or "timeout" in exc.__class__.__name__.lower()
                or "timed out" in message.lower()
            )
            if not is_transient or attempt >= len(retry_delays):
                if status_code == 504 or "504" in message or "DEADLINE_EXCEEDED" in message:
                    raise RuntimeError(
                        "Gemini 분석 요청이 서버 처리 제한 시간 안에 끝나지 않았습니다(504). "
                        "요청 내용을 줄여 재시도했지만 실패했습니다. 잠시 후 다시 시도해 주세요."
                    ) from exc
                if status_code == 503 or "503" in message or "UNAVAILABLE" in message:
                    raise RuntimeError(
                        "Gemini 모델이 현재 요청을 처리할 수 없습니다(503). 잠시 후 다시 시도해 주세요."
                    ) from exc
                if "timeout" in exc.__class__.__name__.lower() or "timed out" in message.lower():
                    raise RuntimeError(
                        "Gemini 응답 대기 시간이 초과되었습니다. 잠시 후 다시 분석해 주세요."
                    ) from exc
                raise
            time.sleep(retry_delays[attempt])

    if response is None or not response.text:
        raise RuntimeError("Gemini가 비어 있는 응답을 반환했습니다.")
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini 응답을 JSON으로 해석하지 못했습니다.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Gemini 응답 형식이 올바르지 않습니다.")
    return data


def extract_claims(client: genai.Client, article: dict, model: str = DEFAULT_MODEL) -> dict:
    prompt = f"""당신은 기업 관련 뉴스의 사실 주장을 분리하는 조사 보조자입니다.
기사에서 검증 가능한 구체적 사실 주장을 최대 4개 추출하세요. 의견·평가·전망은 사실 주장과 구분하세요. 기사에 없는 주장을 추가하지 마세요. 각 사실 주장에는 기업명, 시점(연도·분기), 지표명, 핵심 수치가 포함된 짧고 구체적인 검색어를 제안하세요. 검색어는 특정 검색엔진에 한정하지 말고 과거 보도자료와 공시도 찾을 수 있도록 작성하세요. 기업명을 자동 식별하세요.
제목: {article.get('title','')}
URL: {article.get('url','')}
본문:
{article.get('body','')[:16000]}
"""
    data = _generate_json(client, model, prompt, CLAIM_SCHEMA, 3500)
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        claims = []
    data["claims"] = [x for x in claims if isinstance(x, dict)][:4]
    data["companies"] = data.get("companies") if isinstance(data.get("companies"), list) else []
    return data


def analyze_with_evidence(client: genai.Client, article: dict, extracted: dict, sources: list[dict], model: str = DEFAULT_MODEL) -> dict:
    evidence_block = "\n\n".join(
        f"[{s['id']}] 제목: {s['title']}\n도메인: {s['domain']} / 게시·수집 시각: {s['date']}\n검색어: {s.get('search_query', '')}\nURL: {s['url']}\n자료 유형: {s.get('text_origin', '확인되지 않음')}\n본문·검색 요약: {(s.get('text') or '본문 자동 추출 실패. 제목과 메타정보만 확보됨.')[:1500]}"
        for s in sources
    ) or "검색 결과를 확보하지 못함. 검증 근거 없음."

    prompt = f"""당신은 기업 관련 뉴스의 정보 신뢰성·타당성을 평가하는 분석 보조자입니다. 기사와 제공된 외부 자료만 근거로 평가하세요.

반드시 지킬 원칙:
- 부정적인 기사라는 이유만으로 허위 또는 편향이라고 판단하지 마세요.
- 기사에 적힌 주장과 외부 근거를 각각 대조하고, 검색 결과가 없으면 거짓이 아니라 평가 보류로 처리하세요.
- 서로 다른 도메인이라도 동일 보도자료의 전재일 수 있습니다. 독립성이 확인되지 않으면 교차 검증 점수를 높이지 마세요.
- 제공된 S1~S8 외의 출처 ID, URL, 숫자, 사실을 만들지 마세요.
- 자료 유형이 '검색 결과 요약'이면 이를 기사 전문처럼 취급하지 말고, 요약에 실제로 명시된 정보만 제한적으로 사용하세요.
- 각 주장을 뒷받침하는 자료가 있으면 source_ids에 해당 출처 ID를 반드시 연결하세요. 자료가 관련 없거나 근거가 되지 않으면 연결하지 마세요.
- 주가·시가총액 변화와 기업의 실제 재무 손실을 구분하세요. 시간상 동시에 발생했다는 이유로 인과관계를 단정하지 마세요.
- 점수 항목은 1.0(충족), 0.5(일부 충족), 0.0(확인된 근거상 미충족), null(정보 부족으로 평가 불가) 중 하나입니다. 점수는 기사가 참일 확률이 아닙니다.
- 확인되지 않은 항목은 null로 두고 이유를 notes에 적으세요.
- 편향 신호는 관찰 가능한 표현과 정보 선택에 한정하며 작성자의 의도를 단정하지 마세요.
- 기업과 소비자·투자자에게 미칠 수 있는 영향은 가능성과 확인된 피해를 구분하세요.

[기사]
제목: {article.get('title','')}
URL: {article.get('url','')}
본문:
{article.get('body','')[:8000]}

[1단계에서 추출한 주장]
{json.dumps(extracted, ensure_ascii=False)}

[자동 수집한 자료]
{evidence_block}

각 주장에 대해 판정, 구체적인 이유, 사용한 출처 ID를 작성하세요. 종합 평가에서는 근거의 범위와 한계를 명시하세요.
"""
    data = _generate_json(client, model, prompt, FINAL_SCHEMA, 5000)
    valid_ids = {s["id"] for s in sources}
    for claim in data.get("claims", []):
        if isinstance(claim, dict):
            claim["source_ids"] = clean_source_ids(claim.get("source_ids"), valid_ids)
    data["reliability"] = clamp_score_map(data.get("reliability"), SCORE_KEYS)
    data["validity"] = clamp_score_map(data.get("validity"), VALIDITY_KEYS)
    data["reliability_score"] = score_index(data["reliability"], SCORE_KEYS)
    data["validity_score"] = score_index(data["validity"], VALIDITY_KEYS)
    data["sources"] = sources
    data["companies"] = data.get("companies") or extracted.get("companies", [])
    return data


def run_full_analysis(api_key: str, article: dict, model: str = DEFAULT_MODEL, progress=None) -> dict:
    if not api_key:
        raise ValueError("Gemini API 키가 설정되지 않았습니다. 성인 운영자가 Streamlit Secrets에 GEMINI_API_KEY를 등록해야 합니다.")
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=300000))

    if progress:
        progress("① Gemini가 기사에서 핵심 사실 주장을 추출하고 있어요.")
    extracted = extract_claims(client, article, model)

    queries = []
    for claim in extracted.get("claims", []):
        if claim.get("type") in ("사실 주장", "해석"):
            # Search both the model's focused query and the original claim.
            # The latter helps recover exact figures/phrases omitted from the query.
            for candidate in (claim.get("search_query", ""), claim.get("claim", "")):
                query = str(candidate).strip()
                if query and query not in queries:
                    queries.append(query)
    if not queries and extracted.get("companies"):
        queries = [str(extracted["companies"][0])]

    if progress:
        progress("② GDELT에서 관련 뉴스 자료를 수집하고 있어요.")
    from evidence_search import collect_evidence
    sources = collect_evidence(
        queries,
        article.get("url", ""),
        progress=progress,
        article_date=article.get("date", ""),
    )

    if progress:
        progress("③ Gemini가 기사 주장과 수집된 근거를 대조하고 있어요.")
    result = analyze_with_evidence(client, article, extracted, sources, model)
    result["extracted_claims"] = extracted.get("claims", [])
    result["article_summary"] = extracted.get("article_summary", "")
    return result
