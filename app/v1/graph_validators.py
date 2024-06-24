from enum import Enum
import json
from fastapi import Form, HTTPException, UploadFile
from pydantic import HttpUrl, BaseModel
from typing import List

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


class Model(str, Enum):
    """
    Model to use for graph creation
    """

    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    GPT_4_TURBO = "gpt-4-turbo"


class PdfURLs(BaseModel):
    urls: List[HttpUrl]


async def parse_urls_form(urls: str = Form(None)):
    if urls:
        try:
            urls_data = json.loads(urls)
            if not isinstance(urls_data, list):
                raise ValueError("URLs must be a list")
            if len(urls_data) == 0:
                raise ValueError("URLs list must not be empty")
            return PdfURLs(urls=urls_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in urls")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return None


async def validate_files(files: List[UploadFile]):
    processed_files = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400, detail=f"File {file.filename} is not a PDF"
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size of 10 MB",
            )

        processed_files.append(file.filename)

    return processed_files
