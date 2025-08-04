# 🧭 스타트업 네비게이터

<br>
<div align="center">
  <h2>🚀 당신의 고민에 딱 맞는 책을 AI가 찾아드립니다! 🚀</h2>
</div>
<br>

# 📋 프로젝트 개요

- **프로젝트명**: 스타트업 네비게이터 (Startup Navigator)
- **목표**: 창업가의 성장 단계와 당면 과제, 그리고 구체적인 고민에 맞는 도서를 AI를 통해 맞춤 추천하여 문제 해결을 돕습니다.
- **주요 기능**:
  - **단계별 정보 수집**: 사용자의 스타트업 성장 단계, 당면 과제, 구체적인 고민을 순차적으로 입력받음
  - **AI 기반 도서 추천 (RAG)**: 사용자의 고민을 벡터화하여 가장 관련성 높은 도서 후보군을 검색(Retrieval)하고, LLM을 통해 최종 추천 도서와 맞춤 추천사를 생성(Generation)
  - **인터랙티브 결과 제공**: 추천된 도서의 정보, AI의 추천 이유, 구체적인 적용 방안, 목차 등을 시각적으로 제공

<br>

# 🛠️ 기술 스택

**Framework** ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

**Language** ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

**Data Handling** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

**AI & Model** ![OpenAI API](https://img.shields.io/badge/OpenAI%20API-412991?style=flat&logo=openai&logoColor=white)

**Collaboration Tools** ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)

<br>
----------------------------------------

# 💻 코드: 스타트업 도서 추천 챗봇

## 📝 프로젝트 설명
- **설명**: Streamlit으로 구현된 단일 페이지 웹 애플리케이션입니다. 사용자의 단계별 입력을 받아 RAG(검색 증강 생성) 파이프라인을 통해 맞춤 도서 추천 결과를 제공합니다.
- **주요 기능**:
  - **사용자 입력 세션 관리**: Streamlit의 `session_state`를 활용하여 사용자의 선택(성장 단계, 과제 등)을 단계별로 저장하고 관리
  - **벡터 유사도 기반 검색**: 사용자의 고민 텍스트를 임베딩하고, 미리 구축된 도서 벡터 DB와 코사인 유사도를 계산하여 가장 관련성 높은 Top-K 도서를 추출
  - **LLM을 통한 동적 콘텐츠 생성**: 검색된 도서 정보를 바탕으로, GPT 모델이 사용자의 상황에 맞는 맞춤 추천사와 적용 방안, 목차 등을 JSON 형식으로 생성
  - **동적 UI 렌더링**: 생성된 JSON 결과를 파싱하여 사용자에게 추천 도서 정보, 커버 이미지, 추천 이유 등을 시각적으로 명확하게 전달

## 🏗️ 모델 및 로직 개요
이 프로젝트는 사전 구축된 벡터 저장소와 OpenAI의 언어 모델을 핵심으로 사용합니다.

| 요소 | 설명 | 세부 모델/라이브러리 |
|-----------|----------------------------------------------------------------|------------------|
| **벡터 저장소** | 도서 정보(이름, 저자, 소개글)와 텍스트 임베딩 벡터 저장 | `vector_store.pkl`, `embeddings_matrix.npy` |
| **임베딩 모델** | 사용자의 고민 텍스트를 벡터로 변환하는 역할 | `text-embedding-3-small` |
| **생성 모델** | 검색된 정보를 바탕으로 최종 추천 내용을 JSON으로 생성 | `gpt-4o-mini` |
| **유사도 계산** | 사용자 고민 벡터와 도서 벡터 간의 관련성 측정 | `numpy` (Cosine Similarity) |


## 📂 주요 함수 및 로직
- `load_vector_store()`: Pickle, Numpy 파일을 로드하여 도서 데이터프레임과 임베딩 행렬을 준비합니다.
- `select_growth_stage()`, `select_challenge()`, `get_user_problem()`: Streamlit 버튼과 `chat_input`을 사용하여 사용자 정보를 순차적으로 수집하고 `session_state`에 저장합니다.
- `get_ai_recommendation()`: **핵심 로직**
  1. 사용자의 고민을 `get_embedding` 함수로 벡터화합니다.
  2. 코사인 유사도를 계산해 가장 관련성 높은 5권의 책 정보를 `retrieved_books_str`로 정리합니다.
  3. 사용자 정보와 검색된 책 정보를 포함한 상세한 프롬프트를 구성합니다.
  4. `gpt-4o-mini` 모델에 JSON 형식의 응답을 요청하여 추천 결과를 `st.session_state.final_recommendation`에 저장합니다.
- `show_final_recommendation()`: `session_state`에 저장된 최종 추천 결과를 바탕으로 `st.columns`, `st.expander` 등을 활용하여 사용자에게 보여줄 최종 페이지를 렌더링합니다.