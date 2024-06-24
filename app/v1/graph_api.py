from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional, Union
from .graph_validators import Model, PdfURLs, parse_urls_form, validate_files


graph_router = APIRouter()

@graph_router.post("/create")
async def create_graph(
    model: Model = Form(..., description="Model to use for graph creation"),
    files: List[UploadFile] = File(default=[], description="PDF files to create the graph"),
    urls: Optional[PdfURLs] = Depends(parse_urls_form)
):
    """
    Create a graph from uploaded pdfs and PDF URLs with the given model
    """    

    if not files and (not urls):
        raise HTTPException(status_code=400, detail="Either files or urls must be provided")

    
    processed_files = []
    processed_urls = []

    if files:
        processed_files = await validate_files(files)

    if urls:
        processed_urls = [str(url) for url in urls.urls]

    return {
        "message": "Graph created successfully",
        "processed_files": processed_files,
        "processed_urls": processed_urls,
        "model": model,
    }
