from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings


def create_embedding(text_chunks: Document, embedding: Embeddings):
    embedded_db = Neo4jVector.from_documents(
        text_chunks, OpenAIEmbeddings()
    )

    return embedded_db

