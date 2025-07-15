import streamlit as st
from datasets import load_dataset
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 환경변수 불러오기
try:
  load_dotenv() 
  api_key = os.getenv("SOLAR_API_KEY")

  OpenAI_client = OpenAI(
      api_key=api_key,
      base_url="https://api.upstage.ai/v1",
  )
except ValueError as e:
  st.warning("SOLAR_API_KEY 환경변수가 없습니다.")

# 데이터셋 불러오기
try:
  dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
except Exception as e:
    st.warning(f"dataset 오류:{e}")

try:
  client = chromadb.PersistentClient(path=r"C:\Users\User\Downloads\sessac\project1") # 크로마 저장 공간(폴더에 저장) -> 나중에도 불러올 수 있음
  # 컬렉션은 임베딩, 문서 및 추가 메타데이터를 저장하는 곳, 이름을 지정하여 컬렉션을 만들 수 있음
  collection = client.get_or_create_collection("news_summary") # 클라이언트 이미 있으면 불러오기, 없으면 만들기
except Exception as e:
    st.warning(f"chromaDB 혹은 collection 오류:{e}")

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def solar_api_news(data_limit: int = 10):
  # 가져올 문서 10개
  for i in range(data_limit):
        doc = dataset['document'][i] # 기사 본문
        title = dataset['title'][i]
        date = dataset['date'][i]
        category = dataset['category'][i]
        
        # 기사 요약하기
        try:
          stream = OpenAI_client.chat.completions.create(
              model="solar-pro",
              messages=[
                {
                  "role": "system",
                  "content": "너는 뉴스 기사를 요약하는 유용한 친구야. 중학생도 이해할 수 있게 한두 줄로 잘 요약하지. 그리고 넌 무조건 반말을 쓰고, 이모지를 가끔 사용해."
                },
                {
                  "role": "user", "content": f"이 기사를 요약해줘{doc}"
                }
              ],
              stream=True,
          )

          summary_answer = ""
          for chunk in stream:
              if chunk.choices[0].delta.content is not None:
                    summary_answer += chunk.choices[0].delta.content # 스트리밍으로 쪼개진 답변 붙임
        # 요약에서 오류나면 건너뛰고 다음 순서 반복문으로
        except Exception as e:
          st.warning(f"{i}번째 기사 요약 중 오류:{e}")
          st.warning("다음 기사 요약으로 넘어갑니다.")
          continue
      
        # 요약 임베딩 및 저장하기
        summary_embedding = embedding_model.encode(summary_answer).tolist()
        doc_id = f"news_summary_{i}" # id명 저장 방식
        
        try:
          collection.add(
                  documents=[summary_answer], # 저장할 텍스트 내용
                  embeddings=[summary_embedding], # 텍스트의 임베딩 벡터
                  metadatas=[{"date": date, "title": title, "category": category, "doc_index": i, "type": "summary"}], # 추가 정보(메타데이터)
                  ids=[doc_id] # 고유 ID
              )
        except Exception as e:
          st.warning(f"ChromaDB에 요약 저장 실패:{e}")

        # 주요 키워드 추출
        response = OpenAI_client.chat.completions.create(
          model="solar-mini",
          messages=[
            {
              "role": "system",
              "content": "너는 글에 있는 주요 키워드를 잘 찾는 기자야. 주요 키워드 1개를 뽑아서 그것만 출력하지. 절대 사족을 붙이지 않아."
            },
            {
              "role": "user", "content": f"{summary_answer} 이 글에 있는 주요 키워드 1개만 뽑아서 말해줘."
            }

          ],
          stream=False,
        )

        # stream을 False로 바꿔서 for 안 씀
        keyword_answer = response.choices[0].message.content # 변수에 답변 받은 키워드 저장

        keyword_embedding = embedding_model.encode(keyword_answer).tolist()
        key_id = f"news_keyword_{i}" # id명 저장 방식

        # 임베딩 및 저장하기
        try:
          collection.add(
                  documents=[keyword_answer], # 저장할 텍스트 내용
                  embeddings=[keyword_embedding], # 텍스트의 임베딩 벡터
                  metadatas=[{"date": date, "title": title, "category": category, "doc_index": i, "type": "keyword"}], # 추가 정보(메타데이터)
                  ids=[key_id] # 고유 ID
              )
        except Exception as e:
          st.warning(f"ChromaDB에 키워드 저장 실패:{e}")

# --------------------------streamlit-------------------------------

st.title("📰기사 요약")

# 날짜별 뉴스 정리
def display_news_by_date():
    st.header("📅 날짜별 뉴스 모아보기")
    # collection에 summary 데이터를 metadatas까지 포함해서 변수에 저장
    all_summary_data = collection.get(where={"type": "summary"}, include=["metadatas"])

    # metadatas에 있는 date들 다 불러와서(리스트컴프리헨션) 중복 제거(set), 다시 list로 만들고(list) 최신순으로 정렬(sorted)
    dates = sorted(list(set(metadata['date'] for metadata in all_summary_data['metadatas'])))
    # 날짜 선택칸
    selected_date = st.selectbox("조회할 날짜를 선택하세요.", options=dates)

    # 선택 됐다면,
    if selected_date:
        st.divider() # 구분선
        st.subheader(f"[{selected_date}] 뉴스 목록")
        # collection에서 가져오는데, summary이면서 selected_date에 포함되는 것만 가져옴($and 연산자)
        news_of_day = collection.get(where={"$and": [{"type": "summary"}, {"date": selected_date}]})
        
        # 선택된 날의 요약뉴스들의 ID 개수만큼 순서대로 반복
        for i in range(len(news_of_day['ids'])):
            # i번째에 있는 데이터의 제목을 가져와서 목록으로 표시
            st.markdown(f"**- {news_of_day['metadatas'][i]['title']}**")
            # i번째에 있는 카테고리 캡션으로 보여줌
            st.caption(f"카테고리: {news_of_day['metadatas'][i]['category']}")
            # i번째에 있는 요약뉴스를 파란 박스 안에 보여줌
            st.info(news_of_day['documents'][i])

# 타임라인별 뉴스 정리
def display_timeline_by_topic():
    st.header("🕓 주제별 뉴스 타임라인")
    # collection에 summary 데이터를 metadatas까지 포함해서 변수에 저장
    all_summary_data = collection.get(where={"type": "summary"}, include=["metadatas"])

    # metadatas에 있는 date들 다 불러와서(리스트컴프리헨션) 중복 제거(set), 다시 list로 만들고(list) 최신순으로 정렬(sorted)
    categories = sorted(list(set(metadata['category'] for metadata in all_summary_data['metadatas'])))
    # 카테고리 정하는 칸
    selected_category = st.selectbox("타임라인을 볼 주제(카테고리)를 선택하세요.", options=categories)

    # 선택 됐다면,
    if selected_category:
        st.divider() # 구분선
        st.subheader(f"[{selected_category}] 타임라인")
        
        # collection에서 가져오는데, summary이면서 selected_category에 포함되는 것만 가져옴($and 연산자)
        category_news = collection.get(where={"$and": [{"type": "summary"}, {"category": selected_category}]})

        # Pandas DataFrame 이용해서 표로 만들기
        df = pd.DataFrame({
            'date': [metadata['date'] for metadata in category_news['metadatas']],   # metadats에서 date들 꺼내서 리스트로 만듦
            'title': [metadata['title'] for metadata in category_news['metadatas']], # metadats에서 title들 꺼내서 리스트로 만듦
            'summary': category_news['documents']
        }).sort_values('date', ascending=False) # date 기준으로 내림차순으로 정렬
        
        st.dataframe(df)

# 슬라이드바
with st.sidebar:
   st.header("⚙️ 설정 및 메뉴")

   # 뉴스 데이터 불러오는 버튼
   if st.button("뉴스 데이터 처리 및 저장", help="API를 사용합니다."):
        with st.spinner('뉴스 데이터 처리 중...'):
            solar_api_news()
   if st.button("데이터 저장공간 초기화", help="DB에 저장된 모든 데이터 삭제"):
        client.delete_collection("news_summary")

   # 구분선 
   st.divider()

   # 메뉴 선택칸 
   menu = st.radio(
        "**메뉴를 선택하세요**",
        ("📅 날짜별 뉴스 모아보기", "🕓 주제별 뉴스 타임라인"),
   )

# 메뉴 실행
if menu == "📅 날짜별 뉴스 모아보기":
    display_news_by_date()

if menu == "🕓 주제별 뉴스 타임라인":
    display_timeline_by_topic()

# collection에 데이터가 없으면 실행
if collection.count() == 0:
    solar_api_news()