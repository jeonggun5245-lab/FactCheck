import os
import re
from datetime import date

import requests
import streamlit as st

st.set_page_config(page_title="TruthCheck AI", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1.8rem;}
.hero {padding: 1.35rem 1.5rem; border-radius: 16px; background: #17324d; color: white;}
.hero p {color: #e5edf5; margin-bottom: 0;}
.note {font-size: .9rem; color: #687786;}
</style>
""", unsafe_allow_html=True)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-astra")

st.markdown(
    '<div class="hero"><h1>🧭 TruthCheck AI</h1>'
    '<p>뉴스의 주장을 분석하고 근거를 비교해, 사용자가 스스로 판단하도록 돕는 도구</p></div>',
    unsafe_allow_html=True,
)
st.write("")
st.info(
    "이 도구는 진실/거짓을 최종 판정하는 심판이 아닙니다. "
    "자료의 출처와 근거, 서로 다른 보도의 일치·불일치, 추가 확인이 필요한 부분을 정리합니다. "
    "AI 분석은 오류가 있을 수 있으므로 원문 자료를 직접 확인해야 합니다."
)


def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")


def search_gdelt(query: str, maxrecords: int = 10):
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "sort": "HybridRel",
        "timespan": "1y",
    }
    response = requests.get(
        GDELT_URL,
        params=params,
        headers={"User-Agent": "TruthCheck-Educational-Prototype/2.0"},
        timeout=25,
    )
    response.raise_for_status()
    return response.json().get("articles", [])


def run_ai_research(article_text: str, article_url: str, question: str, company: str):
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    prompt = f"""
너는 뉴스 리터러시와 기업 정보 검증을 돕는 조사 보조자다. 사용자가 제공한 기사를 검토하고 웹 검색으로 독립적인 근거를 확인하라.

[사용자가 제공한 기사 원문/발췌]
{article_text[:14000]}

[기사 URL(참고용)] {article_url or '제공되지 않음'}
[사용자가 확인하려는 질문/주장] {question or '기사의 핵심 주장을 추출해 검토'}
[관련 기업] {company or '지정되지 않음'}

반드시 웹 검색을 활용하고, 출처를 클릭할 수 있는 URL과 함께 제시하라. 가능하면 다음을 구분해 찾아라.
- 기업·기관의 공식 발표, 공시, 정부·공공기관 원자료
- 서로 독립적인 언론사의 보도
- 기존 팩트체크 결과나 학술·전문 자료

다음 순서와 제목으로 한국어 보고서를 작성하라.
1. 기사 요약: 핵심 내용을 3문장 이내로 정리
2. 핵심 주장 추출: 검증 가능한 사실 주장과 의견·전망을 구분
3. 주장별 근거 확인: 각 주장마다 '확인됨 / 일부 확인됨 / 근거 부족 / 자료 간 불일치 / 확인 불가' 중 하나로 표시하고, 판단 이유와 출처를 제시
4. 정보 문제 유형: 과장, 축소, 맥락 누락·왜곡, 조작, 출처 사칭, 사실과 의견 혼합, 해당 없음, 판단 불가 중 해당 항목만 선택. 근거가 없으면 유형을 단정하지 말 것
5. 출처 비교: 출처의 성격, 발표 시점, 원자료 여부, 서로 일치하거나 다른 부분
6. 종합 판단: '현재 자료로는 신뢰 가능 / 추가 확인 필요 / 신뢰하기 어려운 근거가 있음' 중 하나를 잠정 평가로 제시하고, 확신도 점수나 가짜 정밀 수치를 만들지 말 것
7. 사용자가 직접 확인할 다음 단계: 확인할 문서·기관·날짜를 구체적으로 제시
8. 한계: 검색되지 않은 자료, 기사 전문 미확보, 인과관계 미확인 등 한계를 명시

중요 원칙:
- 검색 결과가 없다는 이유로 거짓이라고 판정하지 말 것.
- 여러 사이트가 같은 보도자료를 재전송한 경우 독립적인 교차 검증으로 세지 말 것.
- 출처의 평판만으로 주장의 참·거짓을 판단하지 말 것.
- 확인된 사실과 추론, 가능성을 분리할 것.
- 주가 하락이나 기업 피해가 뉴스 때문에 발생했다고 인과적으로 단정하지 말 것.
- 검색으로 확인한 정보마다 출처 링크를 표시하고, 출처를 만들어내지 말 것.
"""
    response = client.responses.create(
        model=DEFAULT_MODEL,
        tools=[{"type": "web_search"}],
        tool_choice="required",
        input=prompt,
    )
    return response.output_text


# Tabs
news_tab, company_tab, guide_tab = st.tabs(["📰 뉴스 신뢰성 분석", "📊 기업 영향 검토", "📘 사용 안내"])

with news_tab:
    st.subheader("1. 분석할 뉴스 입력")
    st.caption("기사 링크만으로는 본문을 정확히 가져오지 못할 수 있어. 기사 본문이나 검토할 부분을 함께 붙여 넣는 것을 권장해.")
    article_url = st.text_input("기사 URL (선택)", placeholder="https://...")
    article_text = st.text_area(
        "기사 본문 또는 확인할 문장",
        placeholder="뉴스의 핵심 문장이나 기사 본문을 붙여 넣으세요.",
        height=210,
        key="article_text",
    )
    question = st.text_input("특히 확인하고 싶은 주장 (선택)", placeholder="예: 이 기업이 제품 안전 문제를 은폐했다는 주장은 사실인가?")
    company = st.text_input("관련 기업·기관명 (선택)", placeholder="예: 기업명")

    key_ready = bool(get_api_key())
    if key_ready:
        st.success("AI 웹 검색 분석이 설정되어 있어요.")
    else:
        st.warning("AI 분석 기능은 아직 연결되지 않았어요. 배포 설정에 API 키를 등록하면 사용할 수 있습니다. 아래의 GDELT 관련 기사 검색은 API 키 없이 이용할 수 있어요.")

    analyze_col, search_col = st.columns(2)
    with analyze_col:
        analyze_clicked = st.button("🧠 AI로 주장·근거 분석", type="primary", use_container_width=True, disabled=not key_ready)
    with search_col:
        gdelt_clicked = st.button("🔎 관련 보도 검색", use_container_width=True)

    if analyze_clicked:
        if len(article_text.strip()) < 30:
            st.warning("분석할 기사 본문이나 핵심 문장을 30자 이상 입력해 주세요.")
        else:
            with st.spinner("웹에서 관련 자료를 확인하고 분석하고 있어요..."):
                try:
                    result = run_ai_research(article_text.strip(), article_url.strip(), question.strip(), company.strip())
                    st.session_state["last_analysis"] = result
                except Exception as exc:
                    st.error(f"AI 분석 중 오류가 발생했어요: {exc}")
    if st.session_state.get("last_analysis"):
        st.subheader("분석 결과")
        st.markdown(st.session_state["last_analysis"])
        st.caption("AI가 생성한 조사 초안입니다. 인용 링크를 직접 열어 원문과 날짜를 확인하세요.")

    if gdelt_clicked:
        query = question.strip() or company.strip() or article_text.strip()[:180]
        if len(query) < 3:
            st.warning("검색할 주장이나 기업명을 입력해 주세요.")
        else:
            with st.spinner("관련 보도를 검색하고 있어요..."):
                try:
                    articles = search_gdelt(query)
                    if not articles:
                        st.warning("검색 결과가 없어요. 검색어를 짧게 바꾸거나 핵심어를 바꿔 보세요.")
                    else:
                        st.subheader(f"관련 보도 {len(articles)}건")
                        st.caption("관련 기사 목록일 뿐, 기사들이 서로 독립적인지 또는 주장이 사실인지를 자동으로 보증하지 않습니다.")
                        for item in articles:
                            title = item.get("title") or "제목 없음"
                            url = item.get("url") or ""
                            domain = item.get("domain") or "출처 도메인 미상"
                            seen = item.get("seendate") or "날짜 정보 없음"
                            if url:
                                st.markdown(f"**[{title}]({url})**")
                            else:
                                st.markdown(f"**{title}**")
                            st.caption(f"{domain} · {seen}")
                            st.divider()
                except requests.RequestException as exc:
                    st.error(f"검색 중 오류가 발생했어요: {exc}")
                except ValueError:
                    st.error("검색 결과를 읽지 못했어요. 잠시 후 다시 시도해 주세요.")

with company_tab:
    st.subheader("2. 기업 관련 경제적 영향 검토")
    st.write("관련 뉴스 전후의 기업 주가와 시장 지표를 비교해 사건과 같은 시기에 나타난 변화를 살펴봅니다.")
    st.warning("주가 변화만으로 뉴스의 인과적 영향을 입증할 수는 없습니다. 실적 발표, 시장 전체 흐름, 금리, 업종 이슈 등 다른 요인을 함께 검토해야 합니다.")
    firm = st.text_input("기업명", key="firm_name")
    event_date = st.date_input("뉴스 또는 사건 발생일", value=date.today(), key="event_date")
    st.caption("아래 값은 거래소·공시·신뢰할 수 있는 금융 자료에서 직접 확인해 입력하세요. 기준 기간을 일관되게 맞추는 것이 중요합니다.")
    p1, p2 = st.columns(2)
    with p1:
        stock_before = st.number_input("기업 주가: 사건 전 기준값", min_value=0.0, value=0.0, step=100.0)
        market_before = st.number_input("시장 지수: 사건 전 기준값", min_value=0.0, value=0.0, step=1.0)
    with p2:
        stock_after = st.number_input("기업 주가: 사건 후 기준값", min_value=0.0, value=0.0, step=100.0)
        market_after = st.number_input("시장 지수: 사건 후 기준값", min_value=0.0, value=0.0, step=1.0)
    data_source = st.text_input("주가·지수 자료 출처 URL", placeholder="거래소 또는 금융정보 제공기관의 자료 링크")
    if st.button("변화율 비교 계산", use_container_width=True):
        if min(stock_before, stock_after, market_before, market_after) <= 0:
            st.warning("기업 주가와 시장 지수의 사건 전·후 값을 모두 입력해 주세요.")
        else:
            stock_return = (stock_after / stock_before - 1) * 100
            market_return = (market_after / market_before - 1) * 100
            relative = stock_return - market_return
            st.markdown(f"**{firm or '해당 기업'} · 기준일 {event_date.isoformat()}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("기업 주가 변화율", f"{stock_return:+.2f}%")
            c2.metric("시장 지수 변화율", f"{market_return:+.2f}%")
            c3.metric("시장 대비 차이", f"{relative:+.2f}%p")
            st.caption("시장 대비 차이는 단순 비교값이며, 통계적으로 검증된 초과수익이나 뉴스의 인과효과를 뜻하지 않습니다.")
            if data_source.strip():
                st.markdown(f"자료 출처: [{data_source}]({data_source})")
            else:
                st.warning("자료 출처 URL이 입력되지 않았습니다. 결과를 보고서에 사용하기 전에 출처를 기록하세요.")
    st.markdown("**함께 확인할 요인**")
    for item in [
        "같은 시기에 발표된 기업 실적·공시·법적 판단이 있었는가?",
        "코스피·코스닥 등 시장 전체도 같은 방향으로 움직였는가?",
        "같은 업종의 다른 기업 주가도 비슷하게 변했는가?",
        "해당 뉴스가 처음 보도된 시점과 주가 변동 시점이 일치하는가?",
        "기업이 실제 피해를 공시했거나 신뢰할 수 있는 자료로 확인했는가?",
    ]:
        st.checkbox(item, key="factor_" + str(abs(hash(item))))

with guide_tab:
    st.subheader("3. 분석 기준과 한계")
    st.markdown("""
**분석 단계**
1. 뉴스 본문에서 검증 가능한 핵심 주장과 의견·전망을 구분합니다.
2. 웹 검색으로 기업·기관의 공식 자료, 독립 언론 보도, 팩트체크 자료를 찾습니다.
3. 각 주장에 대해 확인 상태와 근거를 정리합니다.
4. 과장·맥락 왜곡·출처 사칭 등 정보 문제 유형을 근거가 있을 때만 분류합니다.
5. 관련 기업의 경제적 영향을 살필 때는 시장 전체 흐름과 다른 사건도 함께 비교합니다.

**판단 표현의 의미**
- 확인됨: 신뢰할 수 있는 원자료 등으로 핵심 내용이 뒷받침됨
- 일부 확인됨: 핵심 내용 중 일부만 근거가 확인됨
- 근거 부족: 현재 확보한 자료만으로 주장을 뒷받침하기 어려움
- 자료 간 불일치: 신뢰할 수 있는 자료들이 서로 다른 내용을 제시함
- 확인 불가: 자료 접근 또는 정보 부족으로 판단할 수 없음

**중요한 한계**
- AI의 요약·분류·검색 결과에는 오류와 누락이 있을 수 있습니다.
- 기사 링크만 입력해도 본문을 자동으로 확보하는 기능은 제공하지 않습니다. 본문을 붙여 넣어야 분석 품질이 좋아집니다.
- 여러 매체가 같은 통신사 기사나 보도자료를 재전송했다면 독립적인 교차 검증이 아닐 수 있습니다.
- 주가 변화는 뉴스 외에도 여러 요인의 영향을 받으므로, 단순 전후 비교로 인과관계를 증명할 수 없습니다.
- AI가 신뢰도 점수를 산출하지 않습니다. 근거와 한계를 제시해 사용자의 판단을 돕는 것이 목적입니다.
""")

st.divider()
st.markdown("**직접 확인할 수 있는 자료**")
resources = [
    ("DART 전자공시시스템", "https://dart.fss.or.kr/"),
    ("한국거래소", "https://www.krx.co.kr/"),
    ("국가통계포털 KOSIS", "https://kosis.kr/"),
    ("공공데이터포털", "https://www.data.go.kr/"),
    ("Google Fact Check Explorer", "https://toolbox.google.com/factcheck/explorer"),
]
for label, url in resources:
    st.markdown(f"- [{label}]({url})")

st.caption("교육용 프로토타입 · 분석 결과는 조사 보조 자료이며, 최종 판단과 인용 전 검증은 사용자의 몫입니다.")
