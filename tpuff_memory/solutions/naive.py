from sentence_transformers import SentenceTransformer
from agents import Agent, function_tool
import pandas as pd

# Create a minilm model
model = SentenceTransformer('all-MiniLM-L6-v2')


DEFAULT_SYSTEM_PROMPT = """
You're being asked to look up information to answer specific questions relating
to agentic memory.

Use your search tool to retrieve the correct answer

Then respond with the answer to the question
"""


def build_agent(corpus: pd.DataFrame, failure_hook) -> Agent:
    indexed_column = None
    dataset = corpus['dataset'].iloc[0]
    if dataset == "longmemevalv2":
        indexed_column = ("Agent Thought: " + corpus["thought"].fillna("")
                          + " Agent Action: " + corpus["action"].fillna("")
                          + " Agent Goal: " + corpus["goal"].fillna(""))
    elif dataset == "amabench":
        indexed_column = "Agent Observation: " + corpus["observation"].fillna("")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    embeddings = model.encode(indexed_column.to_numpy(),
                              show_progress_bar=True, convert_to_numpy=True)

    @function_tool(failure_error_function=failure_hook,
                   timeout=60)
    def search_memories(query: str):
        """Return 5 similar agent events tto query to help answer the question."""
        print(f"Searching for memories related to query: {query}")
        query_embedding = model.encode([query])[0]
        scores = embeddings @ query_embedding

        top_indices = scores.argsort()[-5:][::-1]

        results = []
        for idx in top_indices:
            row = corpus.iloc[idx]
            result = row.to_dict()
            results.append(result)
        return results

    agent = Agent(
        name="Memory Agent",
        instructions=DEFAULT_SYSTEM_PROMPT,
        tools=[search_memories]
    )
    return agent
