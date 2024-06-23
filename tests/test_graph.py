import pytest
import os
from langchain_community.graphs.graph_document import Relationship
from langchain_community.graphs.graph_document import Node

SAMPLE_PDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sample", "Mango.pdf")
)


@pytest.fixture(scope="module")
def sample_pdf_file():
    assert os.path.exists(
        SAMPLE_PDF_PATH
    ), f"Sample PDF file not found at {SAMPLE_PDF_PATH}"
    return SAMPLE_PDF_PATH


def test_graph_document():
    from app.graph import create_graph
    from app.text_parser import chunk_pdf

    text_chunk = chunk_pdf(SAMPLE_PDF_PATH)
    text_chunk = text_chunk[:2]

    graph_document = create_graph(
        text_chunk
    )

    assert isinstance(graph_document, list), "Result should be a list"

    assert all(
        isinstance(doc.nodes, Node) for doc in graph_document[0]
    ), "All graph_document.nodes items should be Node objects"
    
    assert all(
        isinstance(doc.relationships, Node) for doc in graph_document[0]
    ), "All graph_document.relationships items should be Document objects"
    