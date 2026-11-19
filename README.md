# FactCheck / TruthCheck AI — Gemini prototype

기존 FactCheck Streamlit 프로젝트를 바탕으로, 기사 URL 또는 본문을 입력하면 Gemini가 핵심 주장을 추출하고 GDELT 검색 결과와 대조하도록 구성한 시제품입니다.

## 파일 구성
- `app.py`: Streamlit 사용자 화면
- `gemini_analysis.py`: Gemini API 호출, 주장 추출 및 근거 대조
- `article_extractor.py`: 공개 기사 URL 및 본문 추출
- `evidence_search.py`: GDELT 관련 기사 검색 및 접근 가능한 본문 수집
- `scoring.py`: 신뢰성·타당성 점수 계산
- `requirements.txt`: 설치 패키지

## Streamlit Community Cloud 설정
1. 기존 FactCheck 저장소의 루트에 ZIP 안의 파일을 업로드합니다.
2. `requirements.txt`를 교체하고, 새 Python 파일을 추가합니다.
3. Streamlit 앱의 Settings → Secrets에 성인 운영자가 관리하는 키를 다음과 같이 등록합니다.

```toml
GEMINI_API_KEY = "실제_API_키"
GEMINI_MODEL = "gemini-3.8-flash"
```

API 키를 GitHub 코드, README, 화면 캡처, 채팅에 넣지 마세요. 키는 Streamlit Secrets에만 보관합니다.

## 분석 흐름
1. 기사 URL에서 본문을 추출하거나 사용자가 본문을 입력합니다.
2. Gemini가 최대 4개의 사실 주장을 추출하고 검색어를 만듭니다.
3. GDELT DOC 2.0에서 최근 3개월의 관련 보도를 최대 8개 수집합니다.
4. 접근 가능한 자료의 본문 발췌와 기사 주장을 Gemini가 대조합니다.
5. 5개 항목으로 신뢰성과 타당성·균형성을 각각 평가하고 Python에서 점수를 계산합니다.
6. 주장별 검증, 편향 신호, 기업 영향, 출처를 포함한 Markdown 보고서를 제공합니다.

## 한계
- GDELT는 뉴스 탐색 도구이며, 정부·법원·기업 공시 전체를 포괄하지 않습니다.
- 접근 제한·유료벽·동적 페이지에서는 기사 본문을 가져오지 못할 수 있습니다.
- 관련 보도는 원자료가 아닐 수 있고, 전재 보도는 독립적인 교차 검증이 아닙니다.
- 주가 하락과 뉴스 사이의 인과관계는 별도 분석 없이는 단정할 수 없습니다.
- 점수는 진실일 확률이 아니라 사전에 정한 평가 기준을 충족하는 정도입니다.
- 자료가 부족하면 평가 보류가 정상적인 결과입니다.

## Gemini 관련 운영 유의사항
Gemini API의 자격, 허용되는 사용 목적, 연령 요건, 대상 이용자 제한, 무료 등급 데이터 처리 조건을 API 운영자가 공식 약관에서 확인해야 합니다. 이 코드는 기술적 시제품이며 Google 약관 적합성을 보증하지 않습니다. 미성년자에게 API 이용 연령 또는 대상 이용자 제한을 우회하도록 안내하지 마세요.

공식 문서:
- https://ai.google.dev/gemini-api/docs
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/gemini-api/terms
- https://googleapis.github.io/python-genai/
- https://www.gdeltproject.org/
