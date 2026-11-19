from __future__ import annotations
from typing import Any

SCORE_LABELS = {
    "source_transparency": "출처의 투명성",
    "evidence_credibility": "근거의 신뢰성",
    "independent_corroboration": "독립적 교차 확인",
    "factual_accuracy": "사실관계의 정확성",
    "timeliness_context": "시의성 및 맥락",
}
VALIDITY_LABELS = {
    "logic": "근거와 결론의 논리적 연결",
    "headline_match": "제목과 본문의 일치",
    "context": "맥락의 충분성",
    "counterevidence": "관련 반대 근거·당사자 입장",
    "fact_opinion_distinction": "사실과 의견의 구분",
}
SCORE_KEYS = list(SCORE_LABELS)
VALIDITY_KEYS = list(VALIDITY_LABELS)


def clamp_score_map(value: Any, keys: list[str]) -> dict[str, float | None]:
    source = value if isinstance(value, dict) else {}
    result: dict[str, float | None] = {}
    for key in keys:
        raw = source.get(key)
        if raw is None or raw == "평가 불가":
            result[key] = None
            continue
        if isinstance(raw, bool):
            result[key] = None
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            result[key] = None
            continue
        result[key] = number if number in (0.0, 0.5, 1.0) else None
    return result


def score_index(items: dict[str, Any], keys: list[str], minimum: int = 3) -> dict[str, Any]:
    values = [items[k] for k in keys if isinstance(items.get(k), (int, float)) and not isinstance(items.get(k), bool) and items[k] in (0.0, 0.5, 1.0)]
    if len(values) < minimum:
        return {"status": "평가 보류", "score": None, "evaluated": len(values), "total": len(keys)}
    return {"status": "평가 완료", "score": round(sum(values) / len(values) * 5, 1), "evaluated": len(values), "total": len(keys)}


def clean_source_ids(ids: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(ids, list):
        return []
    return list(dict.fromkeys(str(x) for x in ids if str(x) in valid_ids))
