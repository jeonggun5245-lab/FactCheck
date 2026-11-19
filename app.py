from __future__ import annotations

from datetime import date, datetime
import streamlit as st

from article_extractor import extract_article
from gemini_analysis import DEFAULT_MODEL, run_full_analysis
from scoring import SCORE_LABELS, VALIDITY_LABELS

st.set_page_config(page_title="FactCheck | TruthCheck AI", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 1.6rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: #17324d; color: white;}
.hero p {color: #e5edf5; margin-bottom: 0;}
.small-note {font-size: .88rem; color: #687786;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🧭 TruthCheck AI</h1><p>기업 관련 뉴스의 주장과 근거를 자동으로 비교·분석하는 정보 검증 보조 도구</p></div>', unsafe_allow_html=True)
st.write("")
st.info("기사의 사실 여부를 AI가 최종 확정하는 서비스가 아닙니다. 주장별 근거와 출처, 평가할 수 없는 부분을 함께 제시해 검토를 돕습니다. 무료 Gemini API 입력은 제품 개선에 활용될 수 있으므로 비공개·개인정보를 입력하지 마세요.")


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


def render_score(title: str, result: dict, notes: dict, labels: dict):
    st.subheader(title)
    if result.get("score") is None:
        st.warning(f"평가 보류 — 평가 가능한 항목 {result.get('evaluated', 0)}/{result.get('total', 5)}개")
    else:
        st.metric(title, f"{result['score']:.1f}/5.0")
        st.caption(f"평가 가능 항목: {result['evaluated']}/{result['total']}개")
    for key, label in labels.items():
        value = result.get("items", {}).get(key) if isinstance(result.get("items"), dict) else None
        note = notes.get(key, "근거 설명 없음") if isinstance(notes, dict) else "근거 설명 없음"
        st.write(f"**{label}** — {value if value is not None else '평가 불가'}")
        st.caption(str(note))


def build_report(article: dict, result: dict) -> str:
    lines = [
        "# TruthCheck AI 분석 보고서",
        f"- 분석 일시: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 기사 제목: {article.get('title', '')}",
        f"- 기사 URL: {article.get('url', '')}",
        f"- 작성자: {article.get('author', '확인되지 않음')}",
        f"- 기사 날짜: {article.get('date', '확인되지 않음')}",
        "",
        "> AI 분석은 조사 보조 결과이며, 중요한 판단과 인용 전에는 원자료를 직접 확인해야 합니다.",
        "", "## 요약", result.get("summary", ""), "", "## 주장별 검증"
    ]
    for claim in result.get("claims", []):
        if not isinstance(claim, dict):
            continue
        lines += [f"### {claim.get('claim', '')}", f"- 유형: {claim.get('type', '')}", f"- 판정: {claim.get('verdict', '')}", f"- 근거: {claim.get('reason', '')}", f"- 출처 ID: {', '.join(claim.get('source_ids', []))}", ""]
    rel = result.get("reliability_score", {})
    val = result.get("validity_score", {})
    lines += ["## 신뢰성", f"- 점수: {rel.get('score') if rel.get('score') is not None else '평가 보류'}", "## 타당성·균형성", f"- 점수: {val.get('score') if val.get('score') is not None else '평가 보류'}", "## 어조·편향", f"- 어조: {result.get('tone', '')}", f"- 편향 지표: {', '.join(result.get('bias_signals', []))}", result.get('bias_explanation', ''), "## 기업·정보 이용자 영향", f"- 기업: {', '.join(result.get('companies', []))}", f"- 소비자·투자자 영향: {result.get('audience_impact', '')}", f"- 기업 영향: {result.get('corporate_impact', '')}", f"- 관찰된 시장 변화: {result.get('observed_market_change', '')}", f"- 확인된 재무적 영향: {result.get('documented_financial_impact', '')}", f"- 인과관계 평가: {result.get('causality_assessment', '')}", "## 참고 자료"]
    for source in result.get("sources", []):
        lines.append(f"- [{source['id']}] [{source['title']}]({source['url']}) — {source['domain']} / {source['date']}")
    lines += ["", "## 한계"] + [f"- {item}" for item in result.get("limitations", [])]
    lines += ["- GDELT 검색 결과는 관련 보도 탐색 자료이며 공식 원자료를 대체하지 않습니다.", "- 서로 다른 도메인의 기사가 독립적인 취재인지 별도 확인이 필요합니다.", "- AI의 오류 가능성이 있으므로 출처 원문을 직접 확인해야 합니다."]
    return "\n".join(lines)


with st.sidebar:
    st.header("분석 설정")
    model = st.text_input("Gemini 모델", value=get_secret("GEMINI_MODEL", DEFAULT_MODEL))
    key_ready = bool(get_secret("GEMINI_API_KEY"))
    st.success("Gemini API 키가 설정되어 있습니다.") if key_ready else st.error("Gemini API 키가 설정되지 않았습니다.")
    st.caption("API 키는 코드나 GitHub에 저장하지 않고 Streamlit Secrets에서 관리합니다.")

st.subheader("분석할 기사 입력")
st.caption("기사 링크 또는 기사 본문을 입력하세요. URL에서 본문 추출이 실패하면 본문을 직접 붙여 넣도록 안내합니다.")
raw_input = st.text_area("기사 URL 또는 본문", height=190, placeholder="https://... 또는 기사 본문을 붙여 넣으세요.")
analyze_clicked = st.button("🔎 기사 자동 분석 시작", type="primary", use_container_width=True, disabled=not key_ready)

if analyze_clicked:
    if len(raw_input.strip()) < 30:
        st.warning("기사 URL 또는 본문을 입력해 주세요.")
    else:
        try:
            with st.spinner("기사 정보를 가져오고 있어요..."):
                article = extract_article(raw_input.strip())
            st.session_state["tc_article"] = article
            progress_box = st.empty()
            def progress(message: str):
                progress_box.info(message)
            with st.spinner("Gemini가 주장과 근거를 분석하고 있어요. 검색 및 분석에 시간이 걸릴 수 있습니다."):
                result = run_full_analysis(get_secret("GEMINI_API_KEY"), article, model.strip() or DEFAULT_MODEL, progress)
            st.session_state["tc_result"] = result
            st.success("분석이 완료되었습니다.")
        except Exception as exc:
            st.error(f"분석 중 문제가 발생했습니다: {exc}")
            st.caption("API 키, 모델 사용 가능 여부, GDELT 응답, 기사 본문 접근 상태를 확인해 주세요.")

if st.session_state.get("tc_result"):
    article = st.session_state["tc_article"]
    result = st.session_state["tc_result"]
    st.divider()
    st.header("분석 결과")
    st.subheader(article.get("title") or "제목 미확인")
    if article.get("url"):
        st.markdown(f"[원문 기사 열기]({article['url']})")
    st.write(result.get("summary", "요약을 생성하지 못했습니다."))

    tab1, tab2, tab3, tab4 = st.tabs(["주장별 검증", "점수 평가", "편향·영향", "참고 자료"])
    with tab1:
        for index, claim in enumerate(result.get("claims", []), 1):
            st.markdown(f"**{index}. {claim.get('claim', '주장 미확인')}**")
            st.write(f"유형: {claim.get('type', '미분류')}　|　판정: **{claim.get('verdict', '평가 보류')}**")
            st.write(claim.get("reason", "근거 설명 없음"))
            linked = [s for s in result.get("sources", []) if s.get("id") in claim.get("source_ids", [])]
            if linked:
                for source in linked:
                    st.markdown(f"- [{source['id']} · {source['title']}]({source['url']}) — {source['domain']}")
            else:
                st.caption("연결된 자동 검색 출처가 없습니다.")
            st.divider()
    with tab2:
        left, right = st.columns(2)
        with left:
            rel = result.get("reliability_score", {})
            if rel.get("score") is None:
                st.warning(f"신뢰성 평가 보류 ({rel.get('evaluated', 0)}/5개 항목 평가 가능)")
            else:
                st.metric("신뢰성", f"{rel['score']:.1f}/5.0")
            for key, label in SCORE_LABELS.items():
                st.write(f"**{label}:** {result.get('reliability', {}).get(key) if result.get('reliability', {}).get(key) is not None else '평가 불가'}")
                st.caption(result.get("reliability_notes", {}).get(key, ""))
        with right:
            val = result.get("validity_score", {})
            if val.get("score") is None:
                st.warning(f"타당성·균형성 평가 보류 ({val.get('evaluated', 0)}/5개 항목 평가 가능)")
            else:
                st.metric("타당성·균형성", f"{val['score']:.1f}/5.0")
            for key, label in VALIDITY_LABELS.items():
                st.write(f"**{label}:** {result.get('validity', {}).get(key) if result.get('validity', {}).get(key) is not None else '평가 불가'}")
                st.caption(result.get("validity_notes", {}).get(key, ""))
        st.caption("점수는 기사가 참일 확률이 아니라, 설정한 평가 기준을 충족하는 정도를 나타내는 점검 지수입니다.")
    with tab3:
        st.write(f"**기사 어조:** {result.get('tone', '판단 보류')}")
        st.write("**편향 가능성 지표:** " + (", ".join(result.get("bias_signals", [])) or "뚜렷한 지표 미확인"))
        st.write(result.get("bias_explanation", ""))
        st.divider()
        st.subheader("기업·정보 이용자 영향")
        st.write("**관련 기업:** " + (", ".join(result.get("companies", [])) or "자동 식별되지 않음"))
        st.write("**소비자·투자자 영향:** " + str(result.get("audience_impact", "자료 부족")))
        st.write("**기업 영향:** " + str(result.get("corporate_impact", "자료 부족")))
        st.write("**관찰된 주가·시가총액 변화:** " + str(result.get("observed_market_change", "확인되지 않음")))
        st.write("**공식 확인 재무 영향:** " + str(result.get("documented_financial_impact", "확인되지 않음")))
        st.write("**인과관계 평가:** " + str(result.get("causality_assessment", "평가 보류")))
        st.divider()
        with st.expander("추가: 확인한 주가·시장 지수 직접 비교"):
            st.caption("현재 버전은 금융 시세를 자동 수집하지 않습니다. 거래소나 신뢰할 수 있는 금융 자료에서 확인한 동일 기준일의 값을 입력해 비교하세요.")
            firm = st.text_input("기업명", key="manual_firm")
            event_date = st.date_input("사건 기준일", value=date.today(), key="manual_event_date")
            c_before, c_after = st.columns(2)
            with c_before:
                stock_before = st.number_input("기업 주가: 사건 전", min_value=0.0, value=0.0, key="manual_stock_before")
                market_before = st.number_input("시장 지수: 사건 전", min_value=0.0, value=0.0, key="manual_market_before")
            with c_after:
                stock_after = st.number_input("기업 주가: 사건 후", min_value=0.0, value=0.0, key="manual_stock_after")
                market_after = st.number_input("시장 지수: 사건 후", min_value=0.0, value=0.0, key="manual_market_after")
            data_source = st.text_input("주가·지수 자료 출처 URL", key="manual_data_source")
            if st.button("변화율 비교 계산", key="manual_compare"):
                if min(stock_before, stock_after, market_before, market_after) <= 0:
                    st.warning("기업 주가와 시장 지수의 사건 전·후 값을 모두 입력해 주세요.")
                else:
                    stock_return = (stock_after / stock_before - 1) * 100
                    market_return = (market_after / market_before - 1) * 100
                    relative = stock_return - market_return
                    st.markdown(f"**{firm or '해당 기업'} · 기준일 {event_date.isoformat()}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("기업 주가 변화율", f"{stock_return:+.2f}%")
                    m2.metric("시장 지수 변화율", f"{market_return:+.2f}%")
                    m3.metric("시장 대비 차이", f"{relative:+.2f}%p")
                    st.caption("단순 비교값이며 통계적으로 검증된 초과수익이나 뉴스의 인과효과를 뜻하지 않습니다.")
                    if data_source.strip():
                        st.markdown(f"자료 출처: [{data_source}]({data_source})")
                    else:
                        st.warning("자료 출처 URL이 없어 보고서에 사용하기 전에 출처를 확인해야 합니다.")
    with tab4:
        if result.get("sources"):
            for source in result["sources"]:
                st.markdown(f"**{source['id']} · [{source['title']}]({source['url']})**")
                st.caption(f"{source['domain']} · {source['date']} · 본문 추출 {'성공' if source.get('text') else '실패(제목·메타정보만 확보)'}")
        else:
            st.warning("관련 자료를 확보하지 못했습니다. 이 경우 사실 검증은 제한적입니다.")
        st.markdown("검색 데이터: [GDELT Project](https://www.gdeltproject.org/)")

    st.download_button("분석 보고서 다운로드 (.md)", build_report(article, result), file_name="TruthCheck_Report.md", mime="text/markdown", use_container_width=True)

st.divider()
st.caption("TruthCheck AI · 정보 검토를 돕는 시제품입니다. 투자 권유나 확정적인 진위 판정을 제공하지 않습니다.")
