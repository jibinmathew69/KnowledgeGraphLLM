from typing import List
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.graphs import Neo4jGraph


def write_graph(graph_documents: List[GraphDocument]):
    """
    Write the graph documents to a Neo4J DB.

    Args:
        graph_documents (List[GraphDocument]): List of GraphDocument objects        
    """

    graph = Neo4jGraph(driver_config={"max_connection_lifetime": 3600})

    graph.add_graph_documents(graph_documents)
    graph.refresh_schema()

    return graph, graph.structured_schema
