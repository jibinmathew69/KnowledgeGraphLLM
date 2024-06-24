from enum import Enum
import json
from fastapi import Form, HTTPException
from pydantic import HttpUrl, BaseModel
from typing import List


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