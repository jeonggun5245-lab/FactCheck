from scoring import SCORE_KEYS, VALIDITY_KEYS, score_index, clamp_score_map, clean_source_ids


def test_full_score():
    values = {key: 1.0 for key in SCORE_KEYS}
    assert score_index(values, SCORE_KEYS)["score"] == 5.0


def test_partial_average_and_hold():
    values = {"logic": 1.0, "headline_match": 0.5}
    result = score_index(values, VALIDITY_KEYS)
    assert result["score"] is None
    assert result["status"] == "평가 보류"


def test_unavailable_is_not_zero():
    result = clamp_score_map({"logic": None, "context": 0.5}, VALIDITY_KEYS)
    assert result["logic"] is None
    assert result["context"] == 0.5


def test_source_id_allowlist():
    assert clean_source_ids(["S1", "S9", "http://fake"], {"S1", "S2"}) == ["S1"]
