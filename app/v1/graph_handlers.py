import os
import uuid
from fastapi import HTTPException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from ..text_parser import chunk_pdf
from ..graph import create_graph, write_graph, disambiguate, graph_qa_chain, graph_answer
from ..embedding import create_embedding, embedding_qa_chain, embedding_answer
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from .graph_validators import Model


def write_files_to_disk(files_contents):
    file_paths = []

    for content in files_contents:
        unique_filename = str(uuid.uuid4()) + ".pdf"
        path = os.path.join("app", "temp_files", unique_filename)

        with open(path, "wb") as file:
            file.write(content)

        file_paths.append(path)

    return file_paths


async def create_graph_handler(file_locations, model):

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
            text_chunks.extend(chunk_pdf(file)[:3]) #TODO: Remove this limit  
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Failed to load pdf files {e}")
    
    graph_documents = []
    batch_size = 3
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        batch_result = await create_graph(batch, llm)
        graph_documents.extend(batch_result)        
        
    
    graph, _ = write_graph(graph_documents)

    disambiguated_graph = disambiguate(graph, llm)

    embedding_llm = OpenAIEmbeddings()
    embedding_db = create_embedding(text_chunks, embedding_llm)

    return {
        "success": True
    }


def get_answer(query: str, model: Model):
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

    graph = Neo4jGraph()

    try:
        embedding_db = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            "Chunk"  
        )
    except Exception as e:
        return {
            "success": False,
            "error": "Failed to load graph, please create a graph"
        }

    graph_chain = graph_qa_chain(llm, graph)
    embedding_chain = embedding_qa_chain(llm, embedding_db)

    graph_result = graph_answer(graph_chain, query)    
    embedding_result = embedding_answer(embedding_chain, query)

    return {
        "success": True,
        "result": {
            "graph": graph_result["result"],
            "embedding": embedding_result["result"]
        }
    }


def delete_graph():
    graph = Neo4jGraph()
    graph.query("MATCH (n) DETACH DELETE n")
    return {
        "success": True
    }
