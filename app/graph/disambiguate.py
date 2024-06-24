from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_core.language_models import BaseChatModel
from typing import List


def _disambiguation_prompt():
    """
    Prompt LLM to disambiguate between entities.
    """

    system_prompt = """
    Act as a entity disambiugation tool and tell me which values reference the same entity. 
    For example if I give you

    Birds
    Bird
    Ant

    You return to me

    Birds, 1
    Bird, 1
    Ant, 2

    As the Bird and Birds values have the same integer assigned to them, it means that they reference the same entity.

    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                ("Perform disambiguation on the following values: \n{input}"),
            ),
        ]
    )


class DisambiguatedNode(BaseModel):
    """Get the node name and id from the collection of nodes"""

    name: str = Field(..., title="Disambiquated name for the entity")
    id: int = Field(..., title="Disambiquated id for the entity")


class DisambiguatedNodeList(BaseModel):
    """Get the node name and id from the collection of nodes"""

    nodes: List[DisambiguatedNode] = Field(..., title="List of disambiguated nodes")


def get_node_names(graph):
    """
    Get the node names from the graph
    """
    node_names = graph.query("MATCH (n) WHERE NOT n:Chunk RETURN n.name as name")
    node_names_list = [name["name"] for name in node_names]
    return node_names_list


def _get_cluster_map(node_clusters):
    cluster_map = {}
    for cluster_name, cluster_id in node_clusters:
        if cluster_id not in cluster_map:
            cluster_map[cluster_id] = cluster_name.capitalize()


def get_cluster(disambiguated_response):
    """
    Get the cluster of nodes
    """
    node_clusters = [
        [node["name"], node["id"]] for node in disambiguated_response.dict()["nodes"]
    ]
    cluster_map = _get_cluster_map(node_clusters)

    mapped_nodes = []
    for cluster_name, cluster_id in node_clusters:
        mapped_nodes.append([cluster_name,cluster_map[cluster_id]])

    return mapped_nodes




def disambiguate(graph: Neo4jGraph, llm: BaseChatModel):
    """
    Disambiguate the nodes in the graph
    """
    node_names = get_node_names(graph)
    disambiguation_prompt = _disambiguation_prompt()

    disambiguation_chain = disambiguation_prompt | llm.with_structured_output(
        DisambiguatedNodeList
    )
    disambiguated_response = disambiguation_chain.invoke(
        {"input": "\n".join(node_names)}
    )

    cluster = get_cluster(disambiguated_response)

