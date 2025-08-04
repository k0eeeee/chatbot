import streamlit as st
import pandas as pd
from openai import OpenAI
import json

# ==============================================================================
# 0. 페이지 기본 설정 (가장 먼저 실행되어야 합니다)
# ==============================================================================
st.set_page_config(page_title="스타트업 네비게이터", page_icon="🧭")

# --- [삭제] 크롤링 파일 import 부분 ---
# 이 부분은 더 이상 필요 없으므로 삭제했습니다.

# --- 도서 표지 이미지 URL ---
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


# --- 데이터 로드 함수 (CSV 읽기 방식) ---
@st.cache_data
def load_data():
    """
    'books_data_new.csv' 파일을 읽어서 Pandas 데이터프레임으로 반환합니다.
    """
    try:
        # CSV 파일을 데이터프레임으로 읽어옵니다.
        df = pd.read_csv('books_data_new.csv')
        return df
    except FileNotFoundError:
        # 'books_data_new.csv' 파일이 없으면 None을 반환합니다.
        return None

# --- 최종 데이터 로드 ---
all_books_df = load_data()

# ==============================================================================
# Streamlit 챗봇 앱 로직
# ==============================================================================

# 페이지 제목
st.title("🧭 스타트업 네비게이터")
st.caption("🚀 당신의 고민에 딱 맞는 책을 AI가 찾아드립니다!")

# 데이터 로드 실패 시, 앱에 에러 메시지를 표시하고 멈춥니다.
if all_books_df is None:
    st.error("도서 데이터 파일(books_data.csv)을 찾을 수 없습니다. 먼저 크롤링 코드를 실행하여 데이터를 생성해주세요.")
    st.stop()

# Session State 초기화
if 'step' not in st.session_state:
    st.session_state.step = 1
    st.session_state.growth_stage = None
    st.session_state.challenge = None
    st.session_state.user_problem = None
    st.session_state.final_recommendation = None

# --- 단계 1: 성장 단계 선택 ---
def select_growth_stage():
    st.info("당신의 스타트업은 현재 어떤 단계에 있나요?")
    stages = ["아이디어 검증", "MVP 개발/초기 고객 확보", "PMF(시장-제품 적합성) 탐색", "스케일업/투자 유치"]
    for stage in stages:
        if st.button(f"**{stage}**"):
            st.session_state.growth_stage = stage
            st.session_state.step = 2
            st.rerun()

# --- 단계 2: 당면 과제 선택 ---
def select_challenge():
    st.info(f"선택한 단계: **{st.session_state.growth_stage}**\n\n이제, 지금 가장 집중하고 있는 과제를 선택해 주세요.")
    challenges = ["비즈니스 모델/전략", "제품/기술", "마케팅/영업", "팀/조직문화", "투자/재무"]
    for challenge in challenges:
        if st.button(challenge):
            st.session_state.challenge = challenge
            st.session_state.step = 3
            st.rerun()

# --- 단계 3: 주관식 고민 입력 ---
def get_user_problem():
    st.info(f"'{st.session_state.challenge}' 과제와 관련하여, 현재 겪고 있는 가장 구체적인 고민이나 질문을 들려주세요.")
    if prompt := st.chat_input("예: 초기 유저 100명을 모으고 싶은데, 광고비 없이 할 수 있는 방법이 궁금해요."):
        st.session_state.user_problem = prompt
        st.session_state.step = 4
        st.rerun()

# --- 단계 4: LLM 호출 및 추천 생성 ---
def get_ai_recommendation():
    # OpenAI API 키가 secrets에 설정되어 있는지 확인합니다.
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("설정된 OpenAI API 키가 없습니다. .streamlit/secrets.toml 파일을 확인해주세요.")
        st.session_state.step = 3
        return

    with st.spinner("AI가 전체 도서 목록과 당신의 고민을 비교 분석 중입니다..."):
        try:
            client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
            
            stage = st.session_state.growth_stage
            challenge = st.session_state.challenge
            user_problem = st.session_state.user_problem

            # 전체 책 목록을 문자열로 변환하여 AI에게 전달합니다.
            book_list_str = ""
            for index, row in all_books_df.iterrows():
                # intro 컬럼에 비어있는 값이 있을 경우를 대비하여 .get() 사용
                intro_text = row.get('intro', '소개 없음')
                book_list_str += f"{index}. **{row.get('name', '이름 없음')}**: {intro_text}\n"

            # 새로운 기획에 맞춘 프롬프트
            prompt_template = f"""
            당신은 스타트업 창업가를 돕는 전문 컨설턴트입니다.

            [사용자 정보]
            - 성장 단계: '{stage}'
            - 당면 과제: '{challenge}'
            - 구체적인 고민: "{user_problem}"

            [전체 도서 목록]
            {book_list_str}

            [미션]
            1. 사용자의 '구체적인 고민'을 '전체 도서 목록'의 책 소개(intro) 내용과 비교하여, 고민 해결에 가장 적합한 책 **단 한 권**을 선택하세요.
            2. 그 책을 추천하는 새로운 추천 이유를 생성해주세요. 이때, 사용자의 '성장 단계'와 '당면 과제' 정보를 반드시 활용하여 더욱 개인화된 조언을 해주세요.
            
            답변은 반드시 아래의 JSON 형식으로만 출력해야 합니다.
            ```json
            {{
              "chosen_book_index": <선택한 책의 인덱스 번호 (0부터 19 사이의 숫자)>,
              "new_reason": "<새롭게 생성한 맞춤 추천 이유>"
            }}
            ```
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_template}],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            chosen_index = result['chosen_book_index']
            new_reason = result['new_reason']

            # AI가 선택한 인덱스가 유효한지 확인하고 처리합니다.
            safe_index = 0
            try:
                safe_index = int(chosen_index)
                if not 0 <= safe_index < len(all_books_df):
                    safe_index = 0 # 범위를 벗어나면 기본값으로 설정
            except (ValueError, TypeError):
                safe_index = 0 # 변환 실패 시 기본값으로 설정
            
            # .iloc으로 데이터프레임에서 특정 행을 선택하고, to_dict()로 딕셔너리로 변환합니다.
            final_book = all_books_df.iloc[safe_index].to_dict()
            final_book['ai_reason'] = new_reason
            
            st.session_state.final_recommendation = final_book
            st.session_state.step = 5

        except Exception as e:
            st.error(f"AI 추천을 받아오는 중 오류가 발생했습니다: {e}")
            st.session_state.step = 3

# --- 단계 5: 최종 결과 보여주기 ---
def show_final_recommendation():
    book = st.session_state.final_recommendation
    
    if book:
        st.success("AI가 당신의 고민을 위해 고른 맞춤 추천 도서입니다!")
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            book_title = book.get('name', '제목 없음')
            st.image(COVER_IMAGES.get(book_title, "https://via.placeholder.com/150?text=No+Cover"), width=150)
                
        with col2:
            st.subheader(f"📖 {book.get('name', '제목 없음')}")
            st.markdown(f"<p style='color: black;'>저자: {book.get('author', '저자 없음')}</p>", unsafe_allow_html=True)
        
        st.markdown("#### 🤔 AI의 맞춤 추천 이유")
        st.info(book.get('ai_reason', '추천 이유를 생성하지 못했습니다.'))
        
        st.markdown("---")
        if st.button("처음부터 다시 시작하기"):
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