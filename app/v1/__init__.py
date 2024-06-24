from fastapi import APIRouter
from .graph_api import graph_router

router = APIRouter()

router.include_router(graph_router)
