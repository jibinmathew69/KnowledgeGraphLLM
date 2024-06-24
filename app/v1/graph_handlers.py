import os
import uuid
from fastapi import HTTPException
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from ..text_parser import chunk_pdf


def write_files_to_disk(files_contents):
    file_paths = []

    for content in files_contents:
        unique_filename = str(uuid.uuid4()) + ".pdf"
        path = os.path.join("app", "temp_files", unique_filename)

        with open(path, "wb") as file:
            file.write(content)

        file_paths.append(path)

    return file_paths


def create_graph_handler(file_locations, model):

    model = model.value    
    if model.startswith("claude"):
        llm = ChatAnthropic(
            model=model,
            temperature=0,
            max_tokens=4096,
            max_retries=2,
        )
    else:
        llm = ChatOpenAI(
            model=model, 
            temperature=0
        )

    text_chunks = []

    try:
        for file in file_locations:
            text_chunks.extend(chunk_pdf(file))        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Failed to load pdf files")

    return len(text_chunks)
