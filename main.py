from datasets import load_dataset
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

# 환경변수 불러오기
try:
  load_dotenv() 
  api_key = os.getenv("SOLAR_API_KEY")

  OpenAI_client = OpenAI(
      api_key=api_key,
      base_url="https://api.upstage.ai/v1",
  )
except ValueError as e:
  print("SOLAR_API_KEY 환경변수가 없습니다.")

# 데이터셋 불러오기
try:
  dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
except Exception as e:
    print("dataset 오류:", e)

try:
  client = chromadb.PersistentClient(path=r"C:\Users\User\Downloads\sessac\project1") # 크로마 저장 공간(폴더에 저장) -> 나중에도 불러올 수 있음
  # 컬렉션은 임베딩, 문서 및 추가 메타데이터를 저장하는 곳, 이름을 지정하여 컬렉션을 만들 수 있음
  collection = client.get_or_create_collection("news_summary") # 클라이언트 이미 있으면 불러오기, 없으면 만들기
except Exception as e:
    print("chromaDB 혹은 collection 오류:", e)

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
          print(f"{i}번째 기사 요약 중 오류:{e}")
          print("다음 기사 요약으로 넘어갑니다.")
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
          print("ChromaDB에 요약 저장 실패:", e)

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
          print("ChromaDB에 키워드 저장 실패:", e)


# # 주요 주제 추출
# from collections import Counter
# counter = Counter(dataset[:10]['category'])
# most_category = counter.most_common(5) # 가장 많은 카테고리 5개

if collection.count() == 0: # collection에 데이터가 없으면
    solar_api_news()
else:
    print("저장된 데이터가 이미 있습니다.")


# print(collection.get(ids=["news_summary_9"], include=['documents', 'metadatas']))
# print(collection.get(ids=["news_keyword_9"], include=['documents', 'metadatas']))

# # collection 지우기
# client.delete_collection("news_summary")
