import requests
from bs4 import BeautifulSoup
import time # 예의를 지키는 크롤링을 위해 time 라이브러리 추가
import pandas as pd

# 크롤링을 위한 코드 리스트
code_list = [7921251,103990890,91868851,11928450,89707566,146284662,148175776,147973900,116255710,145757238,120242691,142637189,150108736,130167416,148063482,147976182,125313295,126338963,123878435,116605554]

#리스트 생성
name = ['린 스타트업', '제로 투 원', '비즈니스 아이디어의 탄생', '기업 창업가 매뉴얼', '아이디어 불패의 법칙', '그냥 하는 사람', '브랜드 창업 마스터', '창업이 막막할 때 필요한 책', '마케팅 설계자', '스타트업 설계자', '브랜드 설계자', '24시간 완성! 챗GPT 스타트업 프롬프트 설계', '투자자는 무엇에 꽂히는가', '스토리 설계자', '스타트업 30분 회계', '세균무기의 스타트업 바운스백', 'VC 스타트업', '스타트업 HR 팀장들', '스타트업 자금조달 바이블', '스타트업 디자인 씽킹']
author = ['에릭 리스', '피터 틸, 블레이크 매스터스', '데이비드 블랜드, 알렉산더 오스터왈더', '스티브 블랭크, 밥 도프', '알베르토 사보이아', '김한균', '이종구', '이건호, 강주현', '러셀 브런슨', '제프 워커', '러셀 브런슨', '박희용', '비드리머 컨설팅 그룹', '짐 에드워즈', '박순웅', '세균무기', '김기영', '강정욱, 김민교, 윤명훈', '이영보, 서은경, 박찬우, 김봉윤, 신상열, 최준호, 이현권, 이윤주, 임정우', '고은희']
intro = []
table = []

# User-Agent 헤더 추가 (봇으로 인식되는 것을 방지)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for i in code_list:
    # yes24 URL 구성
    url = f'https://www.yes24.com/product/goods/{i}'

    try:
        # HTTP GET 요청을 통해 데이터 가져오기
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # BeautifulSoup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # [수정] 새로운 클래스 이름으로 소개글 요소를 찾습니다.
        intro_element = soup.find('div', class_="infoWrap_txtInner")

        # 소개글 요소가 존재하는지 확인 후 리스트에 추가합니다.
        if intro_element:
            # \n(줄바꿈)을 띄어쓰기로 바꾸고, \r(커서이동)은 완전히 삭제한 후, 양 끝 공백을 제거합니다.
            clean_text = intro_element.text.replace('\n', '').replace('\r', '').strip()
            intro.append(clean_text)
                
        else:
            intro.append("소개 정보를 찾을 수 없습니다.")

    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류 (코드: {i}): {e}")
        intro.append("페이지를 불러올 수 없습니다.")
        
    # 서버에 부담을 주지 않기 위해 각 요청 사이에 짧은 지연시간을 둡니다.
    time.sleep(1)

code_list_table = [7921251,103990890,91868851,11928450,146284662,147973900,116255710,145757238,120242691,142637189,150108736,148063482,147976182,125313295,126338963,123878435,116605554]

for i in code_list_table:
    # yes24 URL 구성
    url = f'https://www.yes24.com/product/goods/{i}'

    try:
        # HTTP GET 요청을 통해 데이터 가져오기
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # BeautifulSoup을 사용하여 HTML 파싱
        soup = BeautifulSoup(response.text, 'html.parser')

        # [수정] 새로운 클래스 이름으로 목차 요소를 찾습니다.
        table_element = soup.find_all('div', class_="infoWrap_txt")[1]

        # 목차 요소가 존재하는지 확인 후 리스트에 추가합니다.
        if table_element:
            #양 끝 공백을 제거합니다.
            clean_text = table_element.text.strip()
            table.append(clean_text)
                
        else:
            table.append("목차 정보를 찾을 수 없습니다.")

    except requests.exceptions.RequestException as e:
        print(f"HTTP 요청 오류 (코드: {i}): {e}")
        table.append("페이지를 불러올 수 없습니다.")
        
    # 서버에 부담을 주지 않기 위해 각 요청 사이에 짧은 지연시간을 둡니다.
    time.sleep(1)

url = f'https://www.yes24.com/product/goods/89707566'

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

table_element = soup.find_all('div', class_="infoWrap_txt")[2]
clean_text = table_element.text.strip()

table.insert(4, clean_text)

url = f'https://www.yes24.com/product/goods/148175776'

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

table_element = soup.find_all('div', class_="infoWrap_txt")[2]
clean_text = table_element.text.strip()

table.insert(6, clean_text)

url = f'https://www.yes24.com/product/goods/130167416'

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, 'html.parser')

table_element = soup.find_all('div', class_="infoWrap_txt")[2]
clean_text = table_element.text.strip()

table.insert(13, clean_text)

data = {
        'name': name,
        'author': author,
        'intro': intro,
        'table': table
    }

df_total = pd.DataFrame(data)
print(df_total)

df_total.to_csv('books_data_new.csv', index=False, encoding='utf-8-sig')