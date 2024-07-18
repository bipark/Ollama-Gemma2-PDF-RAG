# Ollama - Gemma2 기반의 PDF RAG 검색 및 요약

이 프로젝트는 PDF 파일을 청크로 분할하고, 이를 SQLite 데이터베이스에 저장하는 Python 스크립트를 포함하고 있습니다. 목적은 PDF 데이터를 RAG(Retrieval-Augmented Generation) 모델을 사용하여 검색하고 요약하는 것입니다. 

LLM은 Local 모델인 Ollama-Gemma2를 사용했습니다. 

## 주요 기능

- **PDF 파일 로드**: `PyPDFLoader`를 사용하여 PDF 파일을 로드합니다.
- **텍스트 분할**: `RecursiveCharacterTextSplitter`를 사용하여 로드된 문서를 청크로 분할합니다. 이 과정에서 청크의 크기와 중복을 조절할 수 있습니다.
- **데이터베이스 저장**: 분할된 청크를 SQLite 데이터베이스에 저장합니다. 각 청크는 고유 ID와 함께 저장됩니다.
- **검색**: RAG 모델을 사용하여 검색을 수행합니다. 검색 결과는 청크 ID와 함께 반환됩니다.
- **요약**: 검색 결과를 요약합니다. 요약된 결과는 사용자에게 반환됩니다.

## 사용법

```bash
python main.py
```
## 실행결과
첨부된 PDF 파일을 요약하는 예시입니다.

```bash
{'query': '문서를 한글로 요약해주세요.', 'result': '이 논문에서는 데이터에 허용 버퍼를 적용하여 정확도를 높이는 방법을 제안합니다. 격자 형태의 기준 데이터와 예측 데이터 사이의 차이가 일정 범위 이내에 있으면 True Positive로, 그렇지 않으면 False Positive 또는 False Negative로 분류합니다. 그림은 Ground Truth와 Prediction을 시각화하고 버퍼를 적용한 TP, FP, FN 계산 과정을 보여줍니다.\n\n저자들은 실험 설계, 모델 구현, 전처리 등 각 단계에 참여하여 스마트 복합 솔루션을 개발했습니다.  \n\n\n'}
```
