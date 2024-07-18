import os
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


def make_data():
    # SQLite 데이터베이스 연결
    conn = sqlite3.connect('pdf_database.db')
    cursor = conn.cursor()

    # 테이블 생성 (존재하지 않는 경우)
    cursor.execute('''CREATE TABLE IF NOT EXISTS pdf_chunks
                    (id INTEGER PRIMARY KEY, content TEXT)''')

    # PDF 파일 로드 및 청크 분할
    loader = PyPDFLoader("deepcrack.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # 청크를 데이터베이스에 저장
    for chunk in chunks:
        cursor.execute("INSERT INTO pdf_chunks (content) VALUES (?)", (chunk.page_content,))
    conn.commit()

    return cursor

if not os.path.exists('pdf_database.db'):
    cursor = make_data()
else:
    conn = sqlite3.connect('pdf_database.db')
    cursor = conn.cursor()

# 임베딩 생성
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 데이터베이스에서 청크 검색 및 FAISS 인덱스 생성
cursor.execute("SELECT content FROM pdf_chunks")
db_chunks = cursor.fetchall()
texts = [chunk[0] for chunk in db_chunks]

# FAISS 벡터 저장소 생성
vectorstore = FAISS.from_texts(texts, embeddings)

# 검색 기능 설정
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Ollama 모델 생성
ollama_gemma2 = Ollama(model="gemma2")

# QA 체인 생성
qa = RetrievalQA.from_chain_type(llm=ollama_gemma2, chain_type="stuff", retriever=retriever)

# 사용자 쿼리에 대한 응답 생성
query = "문서를 한글로 요약해주세요."
response = qa.invoke(query)

print(response)
