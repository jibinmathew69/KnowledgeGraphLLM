import pytest
import os
from app.text_parser import chunk_pdf
from langchain_core.documents.base import Document

SAMPLE_PDF_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'samples', 'mango.pdf'))

@pytest.fixture(scope="module")
def sample_pdf_file():
    assert os.path.exists(SAMPLE_PDF_PATH), f"Sample PDF file not found at {SAMPLE_PDF_PATH}"
    return SAMPLE_PDF_PATH


def test_parse_pdf():
    text_chunks = chunk_pdf(SAMPLE_PDF_PATH)

    assert isinstance(text_chunks, list), "Result should be a list"

    assert all(isinstance(doc, Document) for doc in text_chunks), "All items should be Document objects"

    assert len(text_chunks) == 7, f"Expected 7 documents, but got {len(text_chunks)}"

    for doc in text_chunks:
        assert doc.page_content.strip() != "", "Document content should not be empty"