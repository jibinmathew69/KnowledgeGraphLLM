from langchain_core.prompts import PromptTemplate

def _get_qa_prompt():
    QA_CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
    Instructions:
    - Use only the provided relationship types and properties in the schema.
    - Do not use any other relationship types or properties that are not listed.
    - Examine the properties of nodes and relationships closely, as the answer might be found there.
    - STRICTLY use **capitalize** when using **name** property of nodes, for example instead of "mango" use "Mango" or "Mango Tree" use "Mango tree".
    - When fetching from properties use OR on few synonyms of values to match, for example instead of just `Leaf` do or on `Leaf` or `Mango leaf`".

    Schema:
    {schema}
    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.

    The question is:
    {question}"""
    return PromptTemplate(
        input_variables=["schema", "question"], template=QA_CYPHER_GENERATION_TEMPLATE
    )

