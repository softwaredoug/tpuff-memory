import asyncio
from agents import Agent, Runner, function_tool
from sentence_transformers import SentenceTransformer
from .ama_bench.dataset import documents


# Create a minilm model
model = SentenceTransformer('all-MiniLM-L6-v2')
corpus = documents("./AMA-bench")


obs_embeddings = model.encode(corpus['observation'].to_numpy(),
                              show_progress_bar=True, convert_to_numpy=True)


@function_tool
def search_memories(query: str):
    """Return 5 similar memories tto query to help answer the question."""
    query_embedding = model.encode([query])[0]
    scores = obs_embeddings @ query_embedding

    top_indices = scores.argsort()[-5:][::-1]

    results = []
    for idx in top_indices:
        row = corpus.iloc[idx]
        result = {
            "action": row['action'],
            "observation": row['observation'],
            "episode_task": row['episode_task'],
        }
        results.append(result)
    return results


DEFAULT_SYSTEM_PROMPT = """
You're being asked to look up information to answer specific questions relating
to agentic memory.

Use your search tool to retrieve the correct answer

Then respond with the answer to the question
"""


async def memory_agent(question: str,
                       system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    agent = Agent(
        name="Memory Agent",
        instructions=system_prompt
    )
    result = await Runner.run(agent, question)
    return result.final_output


if __name__ == "__main__":
    question = "What is the capital of France?"
    answer = asyncio.run(memory_agent(question))
    print(f"Question: {question}")
    print(f"Answer: {answer}")
