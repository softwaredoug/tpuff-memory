from sentence_transformers import SentenceTransformer
from agents import Agent, function_tool
import pandas as pd
from ..facts import extract_all
import asyncio
from pydantic import BaseModel, Field

# Create a minilm model
model = SentenceTransformer('all-MiniLM-L6-v2')


DEFAULT_SYSTEM_PROMPT = """
You're being asked to look up information to answer specific questions relating
to agentic memory.

Use your search tool to retrieve the correct answer

Then respond with the answer to the question
"""


class EntityLookupResults(BaseModel):
    """All context for confusing terminology mentioned in the question."""
    entity: str = Field(..., description="The entity being looked up")
    related_context: list[dict] = Field(..., description="All context for the entity being looked up")


def build_agent(corpus: pd.DataFrame, failure_hook) -> Agent:
    dataset = corpus['dataset'].iloc[0]
    if dataset != "longmemevalv2":
        raise ValueError("Only longmemevalv2 dataset is supported for entity extraction.")

    doc_ids_to_entities = asyncio.run(extract_all(corpus))
    # Init all corpus to empty list for entities
    corpus['entities'] = []
    all_entities = []
    for doc_id, entities in doc_ids_to_entities.items():
        corpus.loc[corpus['doc_id'] == doc_id, 'entities'] = set(entities)
        all_entities.extend(entities)

    # Embed the entities with minilm
    embeddings = model.encode(all_entities,
                              show_progress_bar=True, convert_to_numpy=True)

    @function_tool(failure_error_function=failure_hook,
                   timeout=60)
    def lookup_entity(term: str) -> EntityLookupResults:
        """Lookup other context this term has been mentioned. Particularly entities you're unsure of"""
        print(f"Searching for memories related to query: {term}")
        query_embedding = model.encode([term])[0]
        scores = embeddings @ query_embedding

        top_indices = scores.argsort()[-5:][::-1]

        results = []
        for idx in top_indices:
            entities = corpus['entities'].iloc[idx]
            rows = corpus[corpus['entities'].apply(lambda x: any(e in entities for e in x))].head(5)
            for _, row in rows.iterrows():
                row = corpus.iloc[idx]
                result = row.to_dict()
                results.append(result)
        return EntityLookupResults(entity=term, related_context=results)

    agent = Agent(
        name="Memory Agent",
        instructions=DEFAULT_SYSTEM_PROMPT,
        tools=[lookup_entity]
    )
    return agent
