from langchain_core.documents.base import Document
from langchain_community.graphs.graph_document import GraphDocument
from langchain_experimental.graph_transformers import LLMGraphTransformer
from typing import List


def create_graph(text_chunks: List[Document]) -> List[GraphDocument]:
    """
    Create a graph from a list of text chunks.

    Args:
        text_chunks (List[Document]): List of Document objects, each representing a text chunk

    Returns:
        List[GraphDocument]: List of GraphDocument objects
    """

    graph_maker = LLMGraphTransformer()
