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
            "evidence_quotes": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "source_id": {"type": "STRING"},
                "quote": {"type": "STRING"},
                "explanation": {"type": "STRING"},
            }, "required": ["source_id", "quote", "explanation"]}},
        }, "required": ["claim", "type", "verdict", "reason", "source_ids", "evidence_quotes"]}},
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
- 자료 유형이 '검색 결과 요약'이어도 검증 자료로 활용할 수 있습니다. 다만 기사 전문과 동일한 깊이의 근거로 취급하지 말고, 요약에 구체적으로 명시된 사실·수치·날짜는 직접적인 보도 내용으로 대조하세요.
- 서로 다른 언론사 2곳 이상이 동일한 구체적 사실이나 수치를 명시하면 '복수 매체의 보도 일치'로 인정하세요. 단, 동일 보도자료 전재 여부나 공통 취재원을 확인할 수 없으면 완전한 독립 검증이라고 단정하지 말고 교차 확인 점수를 0.5로 평가할 수 있습니다.
- 검색 요약이라는 이유만으로 사실관계 정확성, 근거 신뢰성, 시의성, 맥락 점수를 null로 만들지 마세요. 수집된 자료에 날짜·수치·사실이 명시되어 있고 복수 매체가 일치하면 실제로 평가 가능한 항목을 평가하세요.
- 자료가 없어서 평가 불가능한 것과 자료가 여러 개 있으나 독립성·세부 산정 근거가 제한적인 것은 구분하세요. 후자는 무조건 평가 보류가 아니라 0.5 등 부분 충족으로 평가하세요.
[출처 연결 규칙]
- 각 주장의 대상·시점·수치·사건을 나누어 출처와 대조하세요. 한 주장에 여러 제품명·인물·수치·기간이 함께 들어 있으면 요소별로 각각 근거를 확인하세요. 예를 들어 제목에 바나나킥만 등장한다면 새우깡·고래밥까지 확인된 것으로 처리하지 마세요. '시가총액 2,640억 원 증가' 보도만으로 '4일 연속 주가 상승'까지 확인된 것으로 처리하지 마세요. 일부 요소만 복수 자료에서 뒷받침되면 그 부분을 명시하고 '일부 확인·맥락 누락'으로 판정하세요.
- 제목이나 같은 기업을 언급한다는 이유만으로 출처를 연결하지 마세요. 핵심 사실을 실제로 뒷받침하는 출처 ID만 source_ids에 넣으세요.
- 출처에 핵심 수치가 없다면 그 수치를 확인하는 근거로 사용하지 마세요. 자료 유형이 '검색 결과 요약'이면 요약에 명시된 내용까지만 근거로 삼으세요.
- 출처가 주장의 일부만 뒷받침하면 확인된 부분과 확인되지 않은 부분을 reason에서 구분하고, 판정도 그 범위에 맞추세요.
- 출처를 연결할 때 reason에 해당 출처 ID(S1 등), 실제 제목 또는 핵심 내용, 확인된 날짜·수치를 구체적으로 적으세요. 출처에 없는 내용을 추정하거나 인용하지 마세요.
- 각 주장마다 evidence_quotes에 근거로 삼은 문장을 원문 그대로 넣으세요. 문장은 제공된 출처 text 안에 실제로 존재해야 하며, 문장을 다듬거나 합치거나 번역하지 마세요. 원문에서 정확히 찾을 수 없는 인용은 넣지 마세요.
- 검색 결과 요약만 있는 경우 그 요약 문장만 인용할 수 있으며, 본문에서 확인한 것처럼 표현하지 마세요.
- 관련 출처가 없으면 source_ids와 evidence_quotes를 빈 배열로 두고 평가 보류 사유를 명시하세요.
- 0.5는 해당 평가 기준을 일부 충족한다는 긍정적 근거가 있을 때만 사용하세요. 근거가 없거나 자료가 부족하면 0.5로 채우지 말고 null로 두세요.
- 관찰된 시장 변화, 소비자·투자자 영향, 기업 영향은 제공 자료에 명시된 내용만 쓰세요. 추정 가능한 일반론은 '가능성'이라고 구분하고, 자료에서 확인되지 않은 시장 언급량·매출·주가 변화를 사실처럼 쓰지 마세요.
- companies에는 기사 본문이나 제공된 출처에 실제 등장하는 기업만 포함하세요.
- 주가·시가총액 변화와 기업의 실제 재무 손실을 구분하세요. 시간상 동시에 발생했다는 이유로 인과관계를 단정하지 마세요.
- 점수 항목은 1.0(충족), 0.5(일부 충족), 0.0(확인된 근거상 미충족), null(정보 부족으로 평가 불가) 중 하나입니다. 점수는 기사가 참일 확률이 아닙니다.
- 점수는 자료가 기사 전문인지 여부만으로 결정하지 말고, 실제로 제공된 모든 자료의 정보량과 일치도를 기준으로 평가하세요. 검색 결과 요약도 제목·매체·날짜·구체적 사실이 명시되어 있으면 평가 근거입니다.
- 신뢰성 점수 기준:
  1) 출처 투명성: 매체명·기사 제목·게시일·접근 가능한 URL이 모두 확인되면 1.0, 일부 정보가 빠지면 0.5, 출처를 식별할 수 없으면 0.0.
  2) 근거의 신뢰성: 기업 공시·거래소·정부 통계·방송 원영상 등 1차 자료로 핵심 사실을 확인하면 1.0. 출처가 식별되는 복수 언론 보도가 구체적인 사실·수치를 일치되게 보도하지만 1차 자료까지 확인하지 못하면 0.5~1.0 사이에서 근거의 구체성에 따라 평가. 제목만 있고 세부 내용이 없는 자료는 0.5 이하. 출처 불명 또는 주장과 무관하면 0.0.
  3) 독립적 교차 확인: 서로 다른 매체·기관 3곳 이상에서 핵심 사실이 일치하면 1.0, 2곳이면 0.5, 1곳이면 0.0. 동일 보도자료 전재 가능성, 공통 취재원 여부를 notes에 설명하고, 도메인 수만으로 독립성을 단정하지 마세요.
  4) 사실관계의 정확성: 주장 속 인물·행동·제품·날짜·수치·기간을 각각 나누어 대조하세요. 모든 핵심 요소가 자료와 일치하면 1.0, 일부만 일치하거나 수치 산정 방식이 확인되지 않으면 0.5, 자료와 모순되면 0.0. 관련 자료가 전혀 없을 때만 null.
  5) 시의성 및 맥락: 사건 발생일·방송일·주가 비교 기간·기사 게시일을 구분해 시간 순서가 명확하면 1.0, 일부 날짜만 확인되면 0.5, 날짜 정보가 없으면 null.
- 타당성·균형성 점수는 기사 본문 자체를 읽고 평가하세요. 외부 자료가 부족하다는 이유만으로 모든 타당성 항목을 null로 만들지 마세요. 각 항목의 notes에는 반드시 (가) 기사에서 확인한 구체적 문장 또는 정보, (나) 그 정보가 평가 기준을 충족하거나 충족하지 못하는 이유, (다) 점수의 근거를 적으세요. '맥락이 부족하다', '논리적 비약이 있다' 같은 일반론만 쓰지 마세요.
  1) 근거와 결론의 논리적 연결: 기사가 제시한 자료·인용·수치가 기사 결론을 실제로 뒷받침하는지 평가. 1.0은 근거와 결론 사이 연결 과정이 제시되고 대안 설명도 검토된 경우. 0.5는 시간적 동시성이나 사례 하나로 결론을 제시해 인과관계가 완전히 입증되지 않은 경우. 0.0은 제시된 근거로 결론을 도출할 수 없거나 근거가 결론과 모순되는 경우.
  2) 제목과 본문의 일치: 제목의 핵심 대상·행동·수치·강도가 본문에서 실제로 설명되는지 평가. 1.0은 제목의 핵심 요소가 본문에서 모두 확인되는 경우. 0.5는 일부 요소만 다루거나 제목이 본문보다 과장된 경우. 0.0은 핵심 내용이 본문에 없거나 제목과 모순되는 경우.
  3) 맥락의 충분성: 독자가 사실의 의미를 판단하는 데 필요한 기준 시점, 비교 대상·기준값, 산정 방식, 관련 시장·사회 상황을 제시했는지 평가. 1.0은 핵심 비교 기준과 다른 주요 변수를 설명한 경우. 0.5는 사건의 전후 관계는 설명하지만 비교 기준이나 다른 변수가 빠진 경우. 0.0은 수치·변화만 제시하고 그것을 해석할 기준을 거의 제공하지 않은 경우.
  4) 관련 반대 근거·당사자 입장: 기사 주장의 신뢰성에 영향을 줄 수 있는 기업·전문가·당사자의 설명과 대안 원인을 다루었는지 평가. 1.0은 주요 반론·대안 설명을 균형 있게 다룬 경우. 0.5는 당사자 입장 또는 대안 설명 중 일부만 다룬 경우. 0.0은 인과관계나 논쟁적 주장을 제시하면서 관련 반론·대안 설명을 전혀 다루지 않은 경우. 반론이 본질적으로 필요하지 않은 단순 사실 전달 기사라면 그 이유를 notes에 밝히고 null 처리할 수 있음.
  5) 사실과 의견의 구분: 수치·사건·발언 같은 검증 가능한 사실과 기자의 평가·추정·홍보성 표현을 구분했는지 평가. 1.0은 사실·해석·의견을 명확히 구분한 경우. 0.5는 일부 평가적 표현이 사실처럼 제시되거나 인용과 사실 서술이 혼재한 경우. 0.0은 의견·추측을 검증된 사실처럼 단정한 경우.
- 타당성 각 항목의 notes는 최소 2문장으로 작성하세요. 첫 문장에는 기사에서 확인한 구체적 근거를 인용하거나 요약하고, 둘째 문장에는 그 근거가 점수에 미친 영향을 설명하세요. 자료에 없는 내용을 만들어 점수를 정당화하지 마세요.
- null은 해당 항목을 판단할 자료가 정말 없을 때만 사용하세요. 근거가 제한적이지만 일부 판단이 가능하면 0.5, 자료에 해당 요소가 없다는 점을 확인할 수 있으면 0.0을 사용하세요. '검색 결과 요약'이라는 이유만으로 null을 주지 마세요.
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

출력 전 각 source_id가 실제 제공된 자료인지, 해당 자료에 주장의 핵심 사실이나 수치가 실제로 포함되어 있는지 확인하세요. 근거가 없거나 간접적이면 억지로 연결하지 마세요. 신뢰성·타당성 점수는 각 기준을 독립적으로 평가하고, 같은 이유를 여러 항목에 복사하지 마세요. 특히 타당성 점수는 실제 기사 제목과 본문에 나타난 문장·수치·설명 구조를 근거로 작성하세요. 각 항목의 notes는 해당 점수를 읽는 사람이 재현할 수 있을 정도로 구체적으로 작성하세요.
"""
    data = _generate_json(client, model, prompt, FINAL_SCHEMA, 5000)
    source_map = {s["id"]: s for s in sources}

    def relevant_excerpt(claim_text: str, source_text: str, limit: int = 520) -> str:
        import re
        raw_terms = re.findall(r"[가-힣A-Za-z0-9]+", claim_text.lower())
        stop = {"있다", "했다", "한다", "대한", "관련", "통해", "이후", "에서", "으로", "하며", "하고", "및", "또는", "대한"}
        terms = {term.replace(",", "") for term in raw_terms if len(term) >= 2 and term not in stop}
        if not source_text.strip():
            return ""
        sentences = [x.strip() for x in re.split(r"(?<=[.!?。])\\s+|(?<=다)\\s+", source_text) if x.strip()]
        if not sentences:
            sentences = [source_text.strip()]
        def score(sentence: str) -> int:
            normalized = sentence.lower().replace(",", "")
            return sum(1 for term in terms if term in normalized)
        ranked = sorted(enumerate(sentences), key=lambda pair: (score(pair[1]), -pair[0]), reverse=True)
        best = ranked[0][1]
        return best[:limit] if score(best) else source_text.strip()[:limit]

    for claim in data.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_text = str(claim.get("claim", ""))
        requested_ids = clean_source_ids(claim.get("source_ids"), set(source_map))
        valid_quotes = []
        for item in claim.get("evidence_quotes", []) if isinstance(claim.get("evidence_quotes"), list) else []:
            if not isinstance(item, dict):
                continue
            source = source_map.get(str(item.get("source_id", "")))
            quote = str(item.get("quote", "")).strip()
            source_text = str(source.get("text", "")) if source else ""
            if source and quote and quote in source_text:
                valid_quotes.append({
                    "source_id": source["id"],
                    "quote": quote,
                    "explanation": str(item.get("explanation", "")),
                })

        # If Gemini selected a relevant source but omitted its quote, attach
        # an exact excerpt from the collected text rather than leaving the
        # user with a bare source ID.
        quoted_ids = {item["source_id"] for item in valid_quotes}
        for source_id in requested_ids:
            source = source_map[source_id]
            source_text = str(source.get("text", "")).strip()
            if source_text and source_id not in quoted_ids:
                excerpt = relevant_excerpt(claim_text, source_text)
                if excerpt:
                    valid_quotes.append({
                        "source_id": source_id,
                        "quote": excerpt,
                        "explanation": "수집된 출처 내용 중 주장과 관련된 문구를 원문 그대로 발췌했습니다.",
                    })
                    quoted_ids.add(source_id)

        # If the model failed to connect sources, recover clearly matching
        # sources by deterministic word/number overlap. This is a retrieval
        # aid; the excerpt itself remains verbatim source text.
        if not valid_quotes:
            import re
            raw_terms = re.findall(r"[가-힣A-Za-z0-9]+", claim_text.lower())
            stop = {"있다", "했다", "한다", "대한", "관련", "통해", "이후", "에서", "으로", "하며", "하고", "및", "또는"}
            terms = {term.replace(",", "") for term in raw_terms if len(term) >= 2 and term not in stop}
            ranked_sources = []
            for source in sources:
                source_text = str(source.get("text", "")).strip()
                searchable = (str(source.get("title", "")) + " " + source_text).lower().replace(",", "")
                overlap = sum(1 for term in terms if term in searchable)
                if source_text and overlap >= 2:
                    ranked_sources.append((overlap, source))
            for _, source in sorted(ranked_sources, key=lambda pair: pair[0], reverse=True)[:4]:
                excerpt = relevant_excerpt(claim_text, str(source.get("text", "")))
                if excerpt:
                    valid_quotes.append({
                        "source_id": source["id"],
                        "quote": excerpt,
                        "explanation": "주장과 겹치는 핵심어가 포함된 수집 자료의 원문 발췌입니다.",
                    })

        claim["evidence_quotes"] = valid_quotes
        claim["source_ids"] = list(dict.fromkeys(item["source_id"] for item in valid_quotes))
        claim["body_evidence_ids"] = sorted({
            item["source_id"] for item in valid_quotes
            if source_map.get(item["source_id"], {}).get("text_origin") == "본문"
        })
        claim["snippet_only_ids"] = sorted({
            item["source_id"] for item in valid_quotes
            if source_map.get(item["source_id"], {}).get("text_origin") == "검색 결과 요약"
        })
        if not claim["source_ids"] and claim.get("verdict") in ("확인", "일부 확인·맥락 누락"):
            claim["verdict"] = "평가 보류"
            claim["reason"] = "수집된 자료에서 이 주장과 연결할 수 있는 구체적인 문구를 찾지 못했습니다."

    data["reliability"] = clamp_score_map(data.get("reliability"), SCORE_KEYS)
    data["validity"] = clamp_score_map(data.get("validity"), VALIDITY_KEYS)
    # Search snippets are still evidence when their contents explicitly
    # support a claim. Assess the collected source set, not only full texts.
    matched_sources = [
        source_map[source_id]
        for claim in data.get("claims", []) if isinstance(claim, dict)
        for source_id in claim.get("source_ids", [])
        if source_id in source_map
    ]
    distinct_publishers = {
        str(source.get("domain", "")).lower()
        for source in matched_sources if source.get("domain")
    }
    if len(distinct_publishers) < 2:
        data["reliability"]["independent_corroboration"] = None
        data["reliability_notes"]["independent_corroboration"] = (
            "주장을 뒷받침하는 서로 다른 매체의 자료가 2곳 미만이어서 교차 확인을 평가하기 어렵습니다."
        )
    elif data["reliability"].get("independent_corroboration") is None:
        data["reliability"]["independent_corroboration"] = 0.5
        data["reliability_notes"]["independent_corroboration"] = (
            f"{len(distinct_publishers)}개 매체의 자료에서 같은 핵심 내용이 확인됩니다. "
            "다만 취재원·원자료의 독립성까지 확인한 것은 아니므로 부분 충족으로 평가합니다."
        )
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
