from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_pdf(pdf_path: str):
    """
    Extracts text from a PDF file and returns a list of Document objects.

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        List[Document]: List of Document objects, each representing a text chunks in the PDF
    """

    pdf_loader = PyPDFLoader(pdf_path)
    if not pdf_loader:
        raise ValueError(f"Invalid PDF file: {pdf_path}")

    docs = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(docs)

    return text_chunks
