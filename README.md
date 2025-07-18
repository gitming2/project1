# 📰 뉴스 요약 챗봇

이 프로젝트는 Streamlit을 활용하여 뉴스 기사를 요약하고, 요약된 내용을 기반으로 질문에 답변하며, 다양한 방식으로 뉴스를 탐색할 수 있는 챗봇 애플리케이션입니다.

Upstage AI의 SOLAR 모델을 사용하여 기사를 요약하고 키워드를 추출하며, ChromaDB를 사용하여 요약 및 키워드를 효율적으로 저장하고 검색합니다.

<br />

## ✏️ 기획 배경

현대 사회에서는 뉴스가 빠르게 생산되고 있지만, 많은 사람들은 뉴스에 대한 흥미 부족, 복잡한 정보 구조, 시간 부족 등의 이유로 뉴스를 회피합니다.

본 프로젝트는 이러한 사용자들에게 뉴스 트렌드와 주요 이슈를 직관적으로 전달하기 위해, matplotlib 기반 시각화와 한 줄 요약 기능을 제공하는 웹 서비스를 기획하였습니다.

뉴스를 “읽기”보다는 “한눈에 파악”할 수 있도록 도와주는 것이 핵심입니다.

## Streamlit app 작동 화면

- 뉴스 요약 챗봇

![Image](https://github.com/user-attachments/assets/4dc0670a-2afe-4881-a809-802380ba9f10)

- 날짜별 뉴스 모아보기

<img width="1277" height="838" alt="Image" src="https://github.com/user-attachments/assets/e895640a-7b7e-42ab-bbae-8ff8ad191c57" />

- 주제별 뉴스 타임라인

<img width="1287" height="845" alt="Image" src="https://github.com/user-attachments/assets/8969a04c-1dbc-4e6d-819d-938569d4cb02" />

- 가장 인기있는 주제

<img width="1277" height="840" alt="Image" src="https://github.com/user-attachments/assets/6acfe858-973a-49d0-9747-38ed0e4ac5d1" />

<br />
<br />

---


<br />

## 🚀 환경 설정

프로젝트를 실행하기 위한 환경 설정 방법입니다.

**1. 의존성 설치**

필요한 라이브러리를 설치합니다.

```Bash
pip install -r requirements.txt
```

**2. API 키 설정**

Upstage AI의 SOLAR 모델을 사용하기 위해 SOLAR_API_KEY를 .env 파일에 설정해야 합니다. 프로젝트 루트 디렉토리에 .env 파일을 생성하고 다음과 같이 내용을 추가합니다.

```.env
SOLAR_API_KEY="당신의_SOLAR_API_키를_여기에_입력하세요"
```

<br />

## 🏗️ 코드 구조

프로젝트의 주요 코드 구조와 각 부분의 역할입니다.

- `app.py` (메인 스크립트):

  - Streamlit 앱의 진입점입니다.

  - 환경 변수 로드 (`dotenv`).

  - Upstage AI (`openai` 클라이언트) 및 ChromaDB 클라이언트 초기화.

  - 뉴스 데이터를 요약하고 ChromaDB에 저장하는 `solar_api_news` 함수.

  - Streamlit 사이드바 및 페이지 설정.

  - 챗봇(`display_chatbot`), 날짜별 뉴스 보기(`display_news_by_date`), 주제별 타임라인(`display_timeline_by_topic`), 인기 주제 분석
  (`display_most_common_category`) 기능을 각각의 함수로 구현.

  - 사용자 선택에 따라 해당 메뉴를 렌더링합니다.

<br />

## ⚙️ 기술 스택

이 프로젝트는 다음과 같은 기술 스택을 활용합니다.

- Frontend & Application Framework: Streamlit

- Large Language Model (LLM): Upstage AI SOLAR(`solar-pro` & `solar-mini`)

  - 뉴스 요약 및 키워드 추출에 사용됩니다.

- Vector Database: ChromaDB

  - 요약된 뉴스 내용 및 키워드의 임베딩을 저장하고 검색하는 데 사용됩니다.

  - `PersistentClient`를 사용하여 데이터를 로컬 파일 시스템에 저장합니다.

- Dataset: daekeun-ml/naver-news-summarization-ko (Hugging Face Datasets)

  - 뉴스 요약을 위한 원본 데이터를 제공합니다.

- Embeddings: `chromadb.utils.embedding_functions.DefaultEmbeddingFunction()`

  - 텍스트 임베딩을 생성합니다. (ChromaDB의 기본 임베딩 함수 사용)

- Data Manipulation: Pandas

- Data Visualization: Matplotlib

<br />

## ▶️ 실행 명령어
프로젝트를 실행하는 방법입니다.
1. 애플리케이션 실행 (다음 명령어를 bash shell에서 실행합니다.)
  ```Bash
  streamlit run app.py
  ```
2. 웹 브라우저에서 자동으로 Streamlit 애플리케이션이 열립니다. 만약 자동으로 열리지 않으면, 터미널에 표시되는 URL을 수동으로 클릭하여 접속합니다.
3. 앱이 실행되면 좌측 사이드바에서 "뉴스 데이터 처리 및 저장" 버튼을 클릭하여 ChromaDB에 뉴스를 요약하고 저장하는 과정을 시작해야 합니다. 이 과정은 SOLAR API를 호출하므로 다소 시간이 걸릴 수 있습니다.
