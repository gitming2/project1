# streamlit 배포
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from datasets import load_dataset
import chromadb
from openai import OpenAI
# from dotenv import load_dotenv
import os
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from chromadb.utils import embedding_functions

# # 환경변수 불러오기
try:
  # load_dotenv() 
  # api_key = os.getenv("SOLAR_API_KEY")
  
  # 해당 코드는 streamlit secrets 사용 시 활성화
  api_key = os.environ["SOLAR_API_KEY"] = st.secrets["SOLAR_API_KEY"]

  OpenAI_client = OpenAI(
      api_key=api_key,
      base_url="https://api.upstage.ai/v1",
  )
except ValueError as e:
  st.warning("SOLAR_API_KEY 환경변수가 없습니다.")

try:
  # 현재 app.py 파일이 있는 디렉토리의 절대 경로를 가져옵니다.
  script_path = os.path.dirname(os.path.abspath(__file__))
  # 데이터베이스 폴더의 절대 경로를 만듭니다.
  db_path = os.path.join(script_path, "chroma")
  
  client = chromadb.PersistentClient(path=db_path) # 크로마 저장 공간(폴더에 저장) -> 나중에도 불러올 수 있음
  # 컬렉션은 임베딩, 문서 및 추가 메타데이터를 저장하는 곳, 이름을 지정하여 컬렉션을 만들 수 있음
  embedding_func = embedding_functions.DefaultEmbeddingFunction()
  collection = client.get_or_create_collection(name="news_summary", embedding_function=embedding_func) # 클라이언트 이미 있으면 불러오기, 없으면 만들기
except Exception as e:
    st.warning(f"chromaDB 혹은 collection 오류:{e}")

def solar_api_news(data_limit: int = 50):
  # 데이터셋 불러오기
  try:
    dataset = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
  except Exception as e:
      st.warning(f"dataset 오류:{e}")
  
  # 가져올 문서 50개
  for i in range(data_limit):
        doc = dataset['document'][i] # 기사 본문
        title = dataset['title'][i]
        date = dataset['date'][i]
        category = dataset['category'][i]
        link = dataset['link'][i]
        
        # 기사 요약하기
        try:
          response = OpenAI_client.chat.completions.create(
              model="solar-pro",
              messages=[
                {
                  "role": "system",
                  "content": "너는 뉴스 기사를 요약하는 유용한 친구야. 중학생도 이해할 수 있게 한두 줄로 잘 요약하지. 그리고 넌 무조건 반말을 쓰고, 이모지를 가끔 사용해."
                },
                {
                  "role": "user", "content": f"이 기사를 요약해줘 {doc}"
                }
              ],
              stream=False,
          )
          summary_answer = response.choices[0].message.content
          
        # 요약에서 오류나면 건너뛰고 다음 순서 반복문으로
        except Exception as e:
          st.warning(f"{i}번째 기사 요약 중 오류:{e}")
          st.warning("다음 기사 요약으로 넘어갑니다.")
          continue
      
        # 요약 임베딩 및 저장하기
        doc_id = f"news_summary_{i}" # id명 저장 방식
        
        try:
          collection.add(
                  documents=[summary_answer], # 저장할 텍스트 내용
                  metadatas=[{"date": date, "title": title, "link": link, "category": category, "doc_index": i, "type": "summary"}], # 추가 정보(메타데이터)
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

        key_id = f"news_keyword_{i}" # id명 저장 방식

        # 임베딩 및 저장하기
        try:
          collection.add(
                  documents=[keyword_answer], # 저장할 텍스트 내용
                  metadatas=[{"type": "keyword"}], # 추가 정보(메타데이터)
                  ids=[key_id] # 고유 ID
              )
        except Exception as e:
          st.warning(f"ChromaDB에 키워드 저장 실패:{e}")

# --------------------------streamlit-------------------------------
# streamlit 사이트 기본 설정
st.set_page_config(
         page_title="뉴스 요약해드립니다!",
         page_icon="📰",
         layout="centered", # 화면 가운데 쪽만 씀
         initial_sidebar_state="auto", # 작은 기기에선 사이드바 숨김
     )

# 뉴스 요약 챗봇
def display_chatbot():
  st.title("💬 뉴스 요약 챗봇")
  st.caption("데이터베이스에 저장된 뉴스 요약본을 바탕으로 질문에 답변합니다.(2022년도 뉴스 데이터)")
  
  # DB에 데이터 없으면,
  if collection.count() == 0:
      st.warning("분석할 데이터가 없습니다. 먼저 데이터 처리를 실행해주세요.")
      return

  # 메세지 초기화
  # "message"라는 대화 기록이 없으면, 빈 리스트 만들기
  if "messages" not in st.session_state: # st.session_state는 사용자의 활동을 기억하는 공간
    st.session_state.messages = [] 

  # 기존 메세지 표시 # 질문할 때마다 이전 메세지 사라지고 새로운 답 뜨는 거 방지
  for message in st.session_state.messages: # 저장된 이전 대화 기록을 하나씩 보여줌
      # st.chat_message 말풍선 모양
      with st.chat_message(message["role"]): # role(user, assistant(봇))에 따라 말풍선 위치 다름
          st.markdown(message["content"]) # 말풍선 안에 실제 대화 넣기
  
  # 대화 최대 길이 설정
  MAX_MESSAGES_BEFORE_DELETION = 12

  # 유저 입력 처리
  # if promtp := ~ -> 입력을 받았다면,
  if prompt := st.chat_input("뉴스에 대해 무엇이 궁금하신가요?"): # 입력창 만듦 # 문자열은 입력칸에 보여짐
    # 이전 대화의 길이 확인
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION: # 대화가 12개가 넘어갈 떄,
        del st.session_state.messages[0] # 제일 옛날에 있던 대화부터 지워줌
        del st.session_state.messages[0]
        
    # 사용자 질문을 대화 기록에 추가하고 화면에 출력
    st.session_state.messages.append({"role": "user", "content": prompt}) # user 질문이 prompt에 저장됨
    with st.chat_message("user"): # 방금 입력한 질문을
        st.markdown(prompt) # user 말풍선으로 보여줌

    # AI 의 답변을 받아서 session state에 저장하고, 보여도 줘야함
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 답변을 담아놓을 공간
        
        # retriever(검색)
        results = collection.query( # collection.query -> 검색해줌
            query_texts=[prompt], # [임베딩된 user 질문]을 기준으로 유사한 것
            n_results=3,                        # 중에 가장 유사한 3개 문서를 가져옴
            where={"type": "summary"} # news_summary에서 요약 타입인 것만
        )
        
        # 검색 결과가 없는 경우 처리
        if not results['documents'][0]:
            response_content = "관련된 뉴스 요약 정보를 찾을 수 없습니다."
        else:
            # 검색된 뉴스 요약을 '참고 자료'로 구성
            # results는 user 질문과 가장 유사한 요약 데이터 문서 3개를 담고 있음
            retrieved_summaries = results['documents'][0] # 첫 번째 질문에 대한 요약들(질문 하나라서 0밖에 없음) [[문서1, 문서2, 문서3]]
            retrieved_metadatas = results['metadatas'][0] # 첫 번째 질문에 대한 메타데이터들 
            
            # 참고자료 만들기
            news_strings = []
            # 요약들과 메타데이터들을 순서대로 엮어줌
            for doc, metadata in zip(retrieved_summaries, retrieved_metadatas):
              news_info = f"- 제목: {metadata['title']}\n- 요약: {doc}"
              # 제목, 요약 넣은 정보 str들을 각각 리스트에
              news_strings.append(news_info)
            # 엔터 두 번으로 구분해서 리스트를 문자열로 합침
            context = "\n\n".join(news_strings)
                
            # 이전 대화 내용 기억하기
            chat_msg = []
            for msg in st.session_state.messages[:-1]: # 현재 말고 이전 대화만 가져오기 위해 -1
              # user 또는 assistant: 대화
              msg_info = f"{msg['role']}: {msg['content']}"
              # 대화를 각각 리스트에
              chat_msg.append(msg_info)
              # 엔터 두 번으로 구분해서 리스트를 문자열로 합침
            chat_history = "\n\n".join(chat_msg)
            
            
            prompt_for_llm = f"""넌 뉴스 기사 전문가야. 아래에 제공된 '참고자료'와 '이전 대화'를 바탕으로 '사용자 질문'에 한국어로 친근하게 답변하지. 
            -무조건 반말을 쓰고 이모지도 가끔 사용해.
            -무조건 '참고 자료'와 '이전 대화'에 있는 내용만을 근거로 답변해. 
            -자료에 없는 내용에 대해서는 '관련 정보를 찾을 수 없습니다.'라고 답변해.
            - [링크]와 같은 불필요한 태그를 절대 포함하지 마.
            
            ---
            [이전 대화]
            {chat_history}
            ---
            [참고자료]
            {context}
            ---
            [사용자 질문]
            {prompt}
            """

            # Solar 모델에 프롬프트를 보내 답변 생성
            try:
                response_content = ""
                stream = OpenAI_client.chat.completions.create(
                    model="solar-pro",
                    messages=[
                        {"role": "system", "content": "너는 주어진 참고자료와 이전 대화를 기반으로 답을 잘 해주는 유용한 친구야."},
                        {"role": "user", "content": prompt_for_llm}
                    ], stream = True # 쪼개서 답함
                )
                
                # stream 처럼 보이게 강제로 설정
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        response_content += chunk.choices[0].delta.content
                        message_placeholder.markdown(response_content + "▌") 
                        
                # 스트리밍이 끝나면 커서(▌)를 제거하고 최종 내용을 다시 한번 업데이트
                message_placeholder.markdown(response_content)
                
            except Exception as e:
                response_content = f"답변 생성 중 오류: {e}"

        # 완성된 챗봇 답변을 화면에 출력하고 대화 기록에 저장
        st.session_state.messages.append({"role": "assistant", "content": response_content})

# 날짜별 뉴스 모아보기
def display_news_by_date():
    st.header("📅 날짜별 뉴스 모아보기")
    # collection에 summary 데이터를 documents, metadatas까지 포함해서 변수에 저장
    all_summary_data = collection.get(where={"type": "summary"}, include=["documents", "metadatas"])
    all_keyword_data = collection.get(where={"type":"keyword"}, include=["documents"])
    
    # 데이터 없으면 함수 탈출
    if collection.count() == 0:
        st.warning("분석할 데이터가 없습니다. 먼저 데이터 처리를 실행해주세요.")
        return

    # metadatas에 있는 date들 다 불러와서(리스트컴프리헨션) 중복 제거(set), 다시 list로 만들고(list) 최신순으로 정렬(sorted) # [:10]은 앞에 10글자만 가져온다는 뜻
    dates = sorted(list(set(metadata['date'][:10] for metadata in all_summary_data['metadatas']))) # metadata는 딕셔너리 # 집합에서 다시 리스트로(index 때문)
    # 날짜 선택칸
    selected_date = st.selectbox("조회할 날짜를 선택하세요.", options=dates)

    # 선택 됐다면,
    if selected_date:
        st.divider() # 구분선
        st.subheader(f"[{selected_date}] 뉴스 목록")
        
        # 요약뉴스들의 ID 개수만큼 순서대로 반복
        for i in range(len(all_summary_data["ids"])):
          # 전체 요약 데이터의 metadatas에 있는 date를 i 순서대로 가져오는데, date가 selected_date로 시작하면
          if all_summary_data["metadatas"][i]["date"].startswith(selected_date):
            # 네모 칸 안에 넣음
            with st.container(border=True):
              # i번째에 있는 데이터의 제목을 가져옴 # markdown = 구문요소 지원(예: 헤더, 볼드, 이탤릭 등)
              st.markdown(f'**• {all_summary_data["metadatas"][i]["title"]}**')
              # 캡션과 링크 버튼을 왼쪽/오른쪽으로 보여주기 위한 columns
              col1, col2 = st.columns([0.7, 0.3]) # 0.7, 0.3 은 나뉠 때 보여지는 비율
              with col1:
                # i번째에 있는 카테고리 캡션으로 보여줌
                st.caption(f'카테고리: {all_summary_data["metadatas"][i]["category"]} | 키워드: {all_keyword_data["documents"][i]}')
              with col2:
                # 누르면 링크로 이동하는 버튼
                st.link_button("뉴스 본문 보러가기", all_summary_data['metadatas'][i]['link'])
              # i번째에 있는 요약뉴스를 파란 박스 안에 보여줌
              st.info(all_summary_data["documents"][i])

# 타임라인별 뉴스 정리
def display_timeline_by_topic():
    st.header("🕓 주제별 뉴스 타임라인")
    # collection에 summary 데이터를 metadatas까지 포함해서 변수에 저장
    all_summary_data = collection.get(where={"type": "summary"}, include=["metadatas"])
    
    # 데이터 없으면 함수 탈출
    if collection.count() == 0:
        st.warning("분석할 데이터가 없습니다. 먼저 데이터 처리를 실행해주세요.")
        return

    # metadatas에 있는 date들 다 불러와서(리스트컴프리헨션) 중복 제거(set), 다시 list로 만들고(list) 최신순으로 정렬(sorted)
    categories = sorted(list(set(metadata['category'] for metadata in all_summary_data['metadatas'])))
    # 카테고리 정하는 칸
    selected_category = st.selectbox("타임라인을 볼 주제를 선택하세요.", options=categories)

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
        
        
        # iterrows로 각 행을 반복 # _(index)는 필요 없어서 비워둠
        for _, row in df.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['title']}**")
                st.caption(row['date'])
                st.write(row['summary'])
                    
        # st.dataframe(df)

# 가장 인기있는 주제
def display_most_common_category():
  st.header("📊 가장 인기있는 주제")
  all_summary_data = collection.get(where={"type": "summary"}, include=["metadatas"])
  
  # 데이터 없으면 함수 탈출
  if collection.count() == 0:
      st.warning("분석할 데이터가 없습니다. 먼저 데이터 처리를 실행해주세요.")
      return
  
  # 중복 제거한 카테고리 리스트
  set_categories = list(set(metadata["category"] for metadata in all_summary_data["metadatas"]))
  # 그냥 카테코리 모아 놓은 리스트
  categories = [metadata["category"] for metadata in all_summary_data["metadatas"]]
  
  # 리스트 컴프리헨션
  # categories = []
  # for metadata in all_summary_data["metadatas"]:
  #   categories.append(metadata["category"])
  
  # categories 카운터
  counter = Counter(categories)
  # 카테고리 top2(카테고리 개수) 보여줌
  most_category = counter.most_common(len(set_categories)) # 예: [('economy', 8), ('IT과학', 2)]
  
  # 파이 그래프로 시각화하기
  ratio = [value[1] for value in most_category] # 파이 그래프의 값
  labels = [value[0] for value in most_category] # 파이 그래프 값의 이름
  
 
  # 리스트 컴프리헨션
  # ratioes = []
  # for value in len(set_categories):
  #   ratioes.append(value[1])
  

  # 스트림릿에서 표시하려면, plt.figure()로 먼저 영역을 잡아주고 st.pyplot() 함수로 사이트에 그려준다.
  fig, ax = plt.subplots(figsize=(8, 8)) # 그래프 크기 정함(가로: 8, 세로: 8) # fig: 도화지, ax: 실제로 그래프가 그려질 x축, y축이 있는 영역
  ax.pie(ratio, labels=labels, autopct='%1.1f%%')
  # streamlit에 그래프 표시
  st.pyplot(fig)
 

# 슬라이드바
with st.sidebar:
  st.header("⚙️ 설정")
  # 뉴스 데이터 불러오는 버튼
  if st.button("뉴스 데이터 처리 및 저장", help="API를 사용합니다."):
    with st.spinner('뉴스 데이터 처리 중...'):
        client.delete_collection("news_summary")
        collection = client.get_or_create_collection(
            name="news_summary",
            embedding_function=embedding_func
        )
        solar_api_news()
    st.session_state.messages = [] # DB 새로 만들었으니 대화 기록 초기화
    st.rerun() # 앱 새로고침 후 변경사항 반영
            
  if st.button("데이터 저장공간 초기화", help="DB에 저장된 모든 데이터 삭제"):
    with st.spinner('데이터베이스를 초기화하는 중...'):
      client.delete_collection(name="news_summary")
      collection = client.get_or_create_collection( # collection 도 새로 만들어서 비우기
              name="news_summary",
              embedding_function=embedding_func
          )
    st.session_state.messages = [] # 대화 기록도 초기화
    st.rerun() # 앱 새로고침 후 변경사항 반영
    
  # 구분선 
  st.divider()
   
  st.header("▶️ 메뉴") 
  # 메뉴 선택칸 
  menu = st.radio(
    "**메뉴를 선택하세요**",
    ("💬뉴스 요약 챗봇", "📅 날짜별 뉴스 모아보기", "🕓 주제별 뉴스 타임라인", "📊 가장 인기있는 주제"),
    )
   
  
   

# 메뉴 실행
if menu == "💬뉴스 요약 챗봇":
  display_chatbot()
elif menu == "📅 날짜별 뉴스 모아보기":
    display_news_by_date()
elif menu == "🕓 주제별 뉴스 타임라인":
    display_timeline_by_topic()
elif menu == "📊 가장 인기있는 주제":
    display_most_common_category()