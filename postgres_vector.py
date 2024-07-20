from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

CONNECTION_STRING = "postgresql+psycopg2://postgres:bk870105@localhost:5432/postgres"

##--------------------------------------------------------------##
def load_pdf_process(file_path):

    # PDF 파일 로드 및 청크 분할
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    return texts

##--------------------------------------------------------------##
def save_to_db(texts, embeddings):
    
    db = PGVector.from_documents(
        documents=texts,
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="pgvector"
    )
    return db

##--------------------------------------------------------------##
def load_from_db(embeddings):

    db = PGVector(
        embedding_function=embeddings,
        collection_name="pgvector",
        connection_string=CONNECTION_STRING,
        use_jsonb=True,
    )

    return db

##--------------------------------------------------------------##
def query_to_db(query, vector_store, llm, chain_type):

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)
    response = qa.invoke(query)

    return response

##--------------------------------------------------------------##
if __name__ == "__main__":
    # PDF 파일 로드 및 텍스트 분할
    texts = load_pdf_process("deepcrack.pdf")

    # 임베딩 생성
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 벡터 저장소 생성
    # vector_store = save_to_db(texts, embeddings)
    vector_store = load_from_db(embeddings)

    # Ollama 모델 생성
    ollama_gemma2 = Ollama(model="gemma2")

    # 사용자 쿼리에 대한 응답 생성
    query = "문서를 한글로 요약해주세요."
    response = query_to_db(query, vector_store, ollama_gemma2, "stuff")

    print(response)
