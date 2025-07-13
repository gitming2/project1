# 각 라이브러리가 있는지 확인 # 터미널에 실행
# pip freeze | grep "transformers"
# pip freeze | grep "datasets"
# 만약 위에서 "datasets" 가 없다면 실행해주세요.
# pip install -Uq datasets

from transformers import pipeline
from datasets import load_dataset

# dataset 가져오기
news_data = load_dataset("daekeun-ml/naver-news-summarization-ko")

print(f"기사 원문:\n{news_data['test']['document'][1]}") # 기사 내용 하나 가져오기

# solar 불러오기
# pip install -Uq openai 터미널에 실행
# dotenv 설치 및 세팅하기
# pip install -q python-dotenv 터미널에 실행

import openai
# openai.__version__

from openai import OpenAI # openai==1.93.1
from dotenv import load_dotenv
import os

# # 실행하면 .env file 내의 환경변수 컴퓨터에 저장
# load_dotenv()

doc = news_data['test']['document'][1]

client = OpenAI(
    api_key= "YOUR_API_KEY",
    base_url="https://api.upstage.ai/v1",
)

stream = client.chat.completions.create(
    model="solar-mini",
    messages=[
      {
        "role": "system",
        "content": "너는 뉴스를 요약해주는 선생님이야. 중학생이 봐도 잘 이해할 정도로 한두 줄로 요약을 해주지. 말투는 친근하고 이모티콘을 부분적으로 사용해. 넌 무조건 한국어, 반말로 대답해. 그리고 매 대답마다 맨 앞에 '🙌이렇게 요약했습니다:'를 꼭 붙여"
      },
      {
        "role": "user",
        "content": f"이 기사를 요약해줘{doc}"
      }

    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")