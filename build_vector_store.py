import pandas as pd
from openai import OpenAI
import numpy as np
import os

# .streamlit/secrets.toml 파일에서 API 키를 로드하기 위한 설정
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path='.streamlit/secrets.toml')
    api_key = os.getenv('OPENAI_API_KEY')
except ImportError:
    print("python-dotenv 라이브러리가 설치되지 않았습니다. pip install python-dotenv 명령어로 설치해주세요.")
    api_key = None

if not api_key:
    raise ValueError("OpenAI API 키를 .streamlit/secrets.toml 파일에 설정해주세요.")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=api_key)
EMBEDDING_MODEL = "text-embedding-3-small"

def get_embedding(text, model=EMBEDDING_MODEL):
    """주어진 텍스트의 임베딩 벡터를 반환합니다."""
    if not isinstance(text, str) or text.strip() == "":
        return None
    text = text.replace("\n", " ")
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"임베딩 생성 중 오류 발생: {e}")
        return None

def build_vector_store():
    """
    books_data_new.csv를 읽어 'intro'와 'table'을 합친 텍스트의 임베딩을 생성하고,
    데이터프레임과 임베딩 행렬을 파일로 저장합니다.
    """
    # 1. 새로운 CSV 파일 로드
    try:
        # [수정] 새로운 CSV 파일 이름을 사용합니다.
        df = pd.read_csv('books_data_new.csv')
        # intro와 table 컬럼의 비어있는 값(NaN)을 빈 문자열로 채웁니다.
        df['intro'] = df['intro'].fillna("")
        df['table'] = df['table'].fillna("")
    except FileNotFoundError:
        print("오류: books_data_new.csv 파일을 찾을 수 없습니다. 파일이 현재 폴더에 있는지 확인해주세요.")
        return

    print("책 소개글과 목차를 합쳐 임베딩 벡터를 생성합니다...")

    # 2. [핵심 수정] intro와 table 텍스트를 하나로 합칩니다.
    # 각 컬럼을 문자열로 변환한 후 합쳐서, 'combined_text'라는 새 컬럼을 만듭니다.
    df['combined_text'] = "책 소개: " + df['intro'].astype(str) + "\n\n목차: " + df['table'].astype(str)
    
    # 3. 합쳐진 텍스트를 기반으로 임베딩을 생성합니다.
    df['embedding'] = df['combined_text'].apply(get_embedding)
    
    # 임베딩 생성에 실패한 행이 있다면 제거합니다.
    df.dropna(subset=['embedding'], inplace=True)

    print("임베딩 생성 완료!")

    # 4. 데이터 저장 (기존과 동일)
    df.to_pickle('vector_store.pkl')
    embeddings_matrix = np.array(df['embedding'].tolist())
    np.save('embeddings_matrix.npy', embeddings_matrix)

    print("✅ 'vector_store.pkl'과 'embeddings_matrix.npy' 파일이 성공적으로 업데이트되었습니다.")
    print("이제 챗봇 앱을 실행할 수 있습니다.")


if __name__ == "__main__":
    build_vector_store()