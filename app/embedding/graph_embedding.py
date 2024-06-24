from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel


def create_embedding(text_chunks: Document, embedding: Embeddings):
    embedded_db = Neo4jVector.from_documents(
        text_chunks, embedding
    )

    return embedded_db

def get_qa_chain(llm: BaseChatModel, db: Neo4jVector):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

def get_answer(embedded_db: Neo4jVector, query: str):
    return embedded_db.invoke(query)
