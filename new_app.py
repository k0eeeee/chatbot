import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import numpy as np
import re # 텍스트 포맷팅을 위해 re 라이브러리 추가

# --- 0. 페이지 기본 설정 ---
st.set_page_config(page_title="스타트업 네비게이터", page_icon="🧭")

# --- 1. 데이터 및 벡터 저장소 로드 ---
@st.cache_data
def load_vector_store():
    """미리 생성된 벡터 데이터베이스 파일을 로드합니다."""
    try:
        df = pd.read_pickle('vector_store.pkl')
        embeddings_matrix = np.load('embeddings_matrix.npy')
        return df, embeddings_matrix
    except FileNotFoundError:
        return None, None

all_books_df, embeddings_matrix = load_vector_store()

# --- OpenAI 클라이언트 초기화 ---
client = None
if "OPENAI_API_KEY" in st.secrets:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    EMBEDDING_MODEL = "text-embedding-3-small"

# --- 유틸리티 함수 ---
def get_embedding(text, model=EMBEDDING_MODEL):
    text = str(text).replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# --- 이미지 URL ---
COVER_IMAGES = {
    '린 스타트업': 'https://image.yes24.com/goods/7921251/XL',
    '제로 투 원': 'https://image.yes24.com/goods/103990890/XL',
    '비즈니스 아이디어의 탄생': 'https://image.yes24.com/goods/91868851/XL',
    '기업 창업가 매뉴얼': 'https://image.yes24.com/goods/11928450/XL',
    '아이디어 불패의 법칙': 'https://image.yes24.com/goods/89707566/XL',
    '그냥 하는 사람': 'https://image.yes24.com/goods/146284662/XL',
    '브랜드 창업 마스터': 'https://image.yes24.com/goods/148175776/XL',
    '창업이 막막할 때 필요한 책': 'https://image.yes24.com/goods/147973900/XL',
    '마케팅 설계자': 'https://image.yes24.com/goods/116255710/XL',
    '스타트업 설계자': 'https://image.yes24.com/goods/145757238/XL',
    '브랜드 설계자': 'https://image.yes24.com/goods/120242691/XL',
    '24시간 완성! 챗GPT 스타트업 프롬프트 설계': 'https://image.yes24.com/goods/142637189/XL',
    '투자자는 무엇에 꽂히는가': 'https://image.yes24.com/goods/150108736/XL',
    '스토리 설계자': 'https://image.yes24.com/goods/130167416/XL',
    '스타트업 30분 회계': 'https://image.yes24.com/goods/148063482/XL',
    '세균무기의 스타트업 바운스백': 'https://image.yes24.com/goods/147976182/XL',
    'VC 스타트업': 'https://image.yes24.com/goods/125313295/XL',
    '스타트업 HR 팀장들': 'https://image.yes24.com/goods/126338963/XL',
    '스타트업 자금조달 바이블': 'https://image.yes24.com/goods/123878435/XL',
    '스타트업 디자인 씽킹': 'https://image.yes24.com/goods/116605554/XL'
}

# --- 페이지 제목 및 데이터 로드 확인 ---
st.title("🧭 스타트업 네비게이터")
st.caption("🚀 당신의 고민에 딱 맞는 책을 AI가 찾아드립니다!")

if all_books_df is None:
    st.error("도서 데이터베이스 파일(vector_store.pkl)을 찾을 수 없습니다. 먼저 build_vector_store.py를 실행해주세요.")
    st.stop()

if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.growth_stage = None
    st.session_state.challenge = None
    st.session_state.user_problem = None
    st.session_state.final_recommendation = None

# --- 단계 1, 2, 3: 사용자 정보 수집 ---
def select_growth_stage():
    st.info("당신의 스타트업은 현재 어떤 단계에 있나요?")
    stages = ["아이디어 검증", "MVP 개발/초기 고객 확보", "PMF(시장-제품 적합성) 탐색", "스케일업/투자 유치"]
    for stage in stages:
        if st.button(f"**{stage}**"):
            st.session_state.growth_stage = stage
            st.session_state.step = 2
            st.rerun()

def select_challenge():
    st.info(f"선택한 단계: **{st.session_state.growth_stage}**\n\n이제, 지금 가장 집중하고 있는 과제를 선택해 주세요.")
    challenges = ["비즈니스 모델/전략", "제품/기술", "마케팅/영업", "팀/조직문화", "투자/재무"]
    for challenge in challenges:
        if st.button(challenge):
            st.session_state.challenge = challenge
            st.session_state.step = 3
            st.rerun()

def get_user_problem():
    st.info(f"'{st.session_state.challenge}' 과제와 관련하여, 현재 겪고 있는 가장 구체적인 고민이나 질문을 들려주세요.")
    if prompt := st.chat_input("예: 초기 유저 100명을 모으고 싶은데, 광고비 없이 할 수 있는 방법이 궁금해요."):
        st.session_state.user_problem = prompt
        st.session_state.step = 4
        st.rerun()

# --- 단계 4: RAG 기반 추천 생성 ---
def get_ai_recommendation():
    if not client:
        st.error("OpenAI API 키가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        st.session_state.step = 3
        return

    with st.spinner("1/2) AI가 당신의 고민과 가장 관련 있는 책들을 찾고 있습니다..."):
        user_problem = st.session_state.user_problem
        query_embedding = get_embedding(user_problem)
        similarities = [cosine_similarity(query_embedding, book_embedding) for book_embedding in embeddings_matrix]
        top_k_indices = np.argsort(similarities)[-5:][::-1]
        
        retrieved_books_str = ""
        for index in top_k_indices:
            book = all_books_df.iloc[index]
            retrieved_books_str += f"- **{book['name']}** (저자: {book['author']}): {book['intro']}\n"

    with st.spinner("2/2) AI가 찾은 정보를 바탕으로 맞춤 추천사를 생성 중입니다..."):
        stage = st.session_state.growth_stage
        challenge = st.session_state.challenge
        
        # [수정] 프롬프트 대폭 업그레이드
        prompt_template = f"""
        당신은 스타트업 창업가를 돕는 세계 최고의 컨설턴트입니다.

        [사용자 정보]
        - 성장 단계: '{stage}'
        - 당면 과제: '{challenge}'
        - 구체적인 고민: "{user_problem}"

        [1차 분석 결과: 가장 관련성 높은 책 후보 목록]
        {retrieved_books_str}

        [최종 미션]
        1. [순위 결정] '후보 목록' 중에서 사용자의 '구체적인 고민'을 해결하는 데 가장 적합한 책을 1, 2, 3순위로 결정하세요.
        2. [근거 강화 리서치] 1순위로 결정된 책에 대해, 추천의 신뢰도를 높일 **구체적인 근거**를 제시하세요. 'OOO 스타트업' 같은 모호한 표현은 절대 사용하지 마세요. 실제 사례를 찾기 어렵다면, 저자의 다른 아티클, 유명인의 긍정적인 리뷰 등을 **가상의 웹 검색 결과**처럼 만들어 근거로 제시하세요.
        3. [목차 검색] 1순위 책의 **실제 목차**를 가상의 웹 검색을 통해 찾아서, 내용에 맞게 줄바꿈(\\n)을 포함한 텍스트로 정리해주세요.
        4. [적용 방향 제안] 검색한 목차를 참고하여, 사용자가 자신의 스타트업에 **어떻게 적용해볼 수 있을지** 구체적인 예시를 2~3가지 제안해주세요. **반드시 각 제안을 "1. ", "2. " 와 같이 숫자로 시작하고 줄바꿈(\\n)으로 구분된 명확한 리스트 형식으로 작성해야 합니다.**
        5. [최종 답변 생성] 위의 모든 정보를 종합하여, 아래 JSON 형식에 맞춰 최종 답변을 생성하세요.

        ```json
        {{
          "best_book": {{
            "title": "<1순위 책 제목>",
            "author": "<1순위 책 저자>"
          }},
          "new_reason": "<근거 강화 리서치 결과를 포함한, 새롭게 생성된 맞춤 추천 이유>",
          "table_of_contents": "<검색으로 찾은, 줄바꿈으로 정리된 목차 텍스트>",
          "application_points": "<숫자 리스트 형식으로 작성된 구체적인 적용 방향 제안>",
          "second_and_third_books": [
            {{
              "title": "<2순위 책 제목>",
              "author": "<2순위 책 저자>"
            }},
            {{
              "title": "<3순위 책 제목>",
              "author": "<3순위 책 저자>"
            }}
          ]
        }}
        ```
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_template}],
                response_format={"type": "json_object"}
            )
            st.session_state.final_recommendation = json.loads(response.choices[0].message.content)
            st.session_state.step = 5
        except Exception as e:
            st.error(f"AI 추천사 생성 중 오류가 발생했습니다: {e}")
            st.session_state.step = 3

# --- 단계 5: 최종 결과 보여주기 (수정) ---
def show_final_recommendation():
    reco = st.session_state.final_recommendation
    
    if reco:
        best_book_info = reco.get('best_book', {})
        best_book_title = best_book_info.get('title')
        
        # 데이터프레임에서 1순위 책의 '소개글' 정보만 가져옵니다. (목차는 AI가 생성)
        best_book_details = all_books_df[all_books_df['name'] == best_book_title].iloc[0]

        st.success("AI가 당신의 고민을 위해 고른 맞춤 추천 도서입니다!")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(COVER_IMAGES.get(best_book_title, "https://via.placeholder.com/150?text=No+Cover"), width=150)
        with col2:
            st.subheader(f"📖 {best_book_title}")
            st.markdown(f"<p style='color: black;'>저자: {best_book_info.get('author')}</p>", unsafe_allow_html=True)
        
        st.markdown("#### 🤔 AI의 맞춤 추천 이유")
        st.info(reco.get('new_reason'))
        
        st.markdown("#### 💡 이 책의 적용 방향 제안")
        # st.markdown을 사용하면 "1. ... \n 2. ..." 와 같은 텍스트가 리스트로 예쁘게 보입니다.
        st.warning(reco.get('application_points'))

        # [수정] Expander 제목 변경 및 내용 포맷팅
        with st.expander("추천 도서 책소개 및 목차 보기"):
            st.markdown("##### 책 소개")
            intro_text = best_book_details.get('intro', '소개 정보 없음')
            # 마침표 뒤에 줄바꿈을 추가하여 문장별로 보이게 함
            formatted_intro = intro_text.replace('. ', '.\n\n')
            st.write(formatted_intro)
            
            st.markdown("##### 목차")
            # AI가 생성해준, 줄바꿈이 포함된 목차 텍스트를 그대로 사용합니다.
            table_text = reco.get('table_of_contents', '목차 정보를 불러오지 못했습니다.')
            st.text(table_text)

        st.markdown("---")
        
        other_books = reco.get('second_and_third_books', [])
        if other_books:
            st.markdown("##### 📚 함께 읽으면 좋은 책들")
            for book in other_books:
                book_title = book.get('title')
                book_author = book.get('author')
                
                col1_other, col2_other = st.columns([1, 5])
                with col1_other:
                    st.image(COVER_IMAGES.get(book_title, "https://via.placeholder.com/75?text=No+Cover"), width=75)
                with col2_other:
                    st.write(f"**{book_title}**")
                    st.write(f"_{book_author}_")

        st.markdown("---")
        if st.button("다른 고민으로 시작하기"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    else:
        st.error("추천 결과를 불러오는 데 실패했습니다. 다시 시도해주세요.")

# --- 메인 로직 ---
if 'step' not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    select_growth_stage()
elif st.session_state.step == 2:
    select_challenge()
elif st.session_state.step == 3:
    get_user_problem()
elif st.session_state.step == 4:
    get_ai_recommendation()
    if st.session_state.step == 5:
        st.rerun()
elif st.session_state.step == 5:
    show_final_recommendation()