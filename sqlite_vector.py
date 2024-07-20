from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import SQLiteVSS


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
    
    doc_func = lambda x: x.page_content
    docs = list(map(doc_func, texts))    

    vector_store = SQLiteVSS.from_texts(
        texts=docs,
        embedding=embeddings,
        table="documents"
    )
    return vector_store

##--------------------------------------------------------------##
def load_from_db(embeddings):
    conn = SQLiteVSS.create_connection(db_file="vss.db")        
    db = SQLiteVSS(table="documents", embedding=embeddings, connection=conn)
#    vector_store = db.from_texts(texts=texts, embedding=embeddings)

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
    # texts = load_pdf_process("deepcrack.pdf")

    # 임베딩 생성
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 벡터 저장소 생성
    # vector_store = save_to_db(texts, embeddings)
    vector_store = load_from_db(embeddings)

    # Ollama 모델 생성
    ollama_gemma2 = Ollama(model="gemma2")

    # 사용자 쿼리에 대한 응답 생성
    query = "문서를 한글로 요약해주세요."
    response = query_to_db(query, vector_store, ollama_gemma2, "stuff")

    print(response)
