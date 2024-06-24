from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
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

