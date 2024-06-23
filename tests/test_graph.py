import pytest
import os
from langchain_community.graphs.graph_document import Relationship
from langchain_community.graphs.graph_document import Node
from langchain_anthropic import ChatAnthropic
from langchain_community.graphs import Neo4jGraph
from app.graph import create_graph
from app.text_parser import chunk_pdf


SAMPLE_PDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sample", "Mango.pdf")
)


@pytest.fixture(scope="module")
def sample_pdf_file():
    assert os.path.exists(
        SAMPLE_PDF_PATH
    ), f"Sample PDF file not found at {SAMPLE_PDF_PATH}"
    return SAMPLE_PDF_PATH


@pytest.fixture(scope="module")
def graph_document():
    text_chunk = chunk_pdf(SAMPLE_PDF_PATH)
    text_chunk = text_chunk[:2]

    return create_graph(
        text_chunk,
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            max_tokens=4096,
            max_retries=2
        )
    )


def test_graph_document(graph_document):    
    assert isinstance(graph_document, list), "Result should be a list"    

    assert all(
        isinstance(nodes, Node) for nodes in graph_document[0].nodes
    ), "All graph_document.nodes items should be Node objects"
    
    assert all(
        isinstance(relationships, Relationship) for relationships in graph_document[0].relationships
    ), "All graph_document.relationships items should be Relationship objects"


