from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import List, Optional
from .graph_validators import Model, PdfURLs, parse_urls_form, validate_files
from .graph_handlers import write_files_to_disk, create_graph_handler
from .graph_handlers import get_answer, delete_graph



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

    file_locations = []

    if files:
        files_content = await validate_files(files)
        file_paths = write_files_to_disk(files_content)
        file_locations.extend(file_paths)

    if urls:
        file_urls = [str(url) for url in urls.urls]
        file_locations.extend(file_urls)
    
    result = await create_graph_handler(file_locations, model)    
    return result


@graph_router.post("/ask")
async def ask(
    query: str = Form(..., description="Query to ask"),
    model: Model = Form(..., description="Model to use for answering the query"),
):
    """
    Get answer from the graph
    """
    
    return get_answer(query, model)


@graph_router.delete("/clear_db")
async def clear_db():
    """
    Clear the graph database
    """
    return delete_graph()    
