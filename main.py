from datasets import load_dataset
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")

import chromadb
client = chromadb.Client() # 크로마 저장 공간
# 컬렉션은 임베딩, 문서 및 추가 메타데이터를 저장하는 곳입니다. 이름을 지정하여 컬렉션을 만들 수 있습니다
collection = client.create_collection("news_summary")

from openai import OpenAI
from dotenv import load_dotenv
import os

# 실행하면 .env file 내의 환경변수 컴퓨터에 저장
load_dotenv()
api_key = os.getenv("SOLAR_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.upstage.ai/v1",
)

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 가져올 문서 10개
for i in range(10):
    doc = dataset['document'][i] # 기사 본문
    title = dataset['title'][i]
    date = dataset['date'][i]

    stream = client.chat.completions.create(
        model="solar-mini",
        messages=[
          {
            "role": "system",
            "content": "너는 뉴스 기사를 요약하는 유용한 친구야. 중학생도 이해할 수 있게 한두 줄로 잘 요약하지. 그리고 넌 무조건 반말을 쓰고, 이모지를 가끔 사용해."
          },
          {
            "role": "user",
            "content": f"이 기사를 요약해줘{doc}"
          }

        ],
        stream=True,
    )

    answer = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
              answer += chunk.choices[0].delta.content # 스트리밍으로 쪼개진 답변 붙임
              print(chunk.choices[0].delta.content, end="")

    summary_embedding = embedding_model.encode(answer).tolist()
    doc_id = f"news_summary_{i}"

    # 임베딩 및 저장하기
    collection.add(
            documents=[answer], # 저장할 텍스트 내용
            embeddings=[summary_embedding], # 텍스트의 임베딩 벡터
            metadatas=[{"date": date, "title": title, "doc_index": i}], # 추가 정보(메타데이터)
            ids=[doc_id] # 고유 ID
        )