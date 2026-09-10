import asyncio
from agents import Agent, Runner, function_tool
from sentence_transformers import SentenceTransformer
from .dataset import load_dataset
import argparse


# Create a minilm model
model = SentenceTransformer('all-MiniLM-L6-v2')


DEFAULT_SYSTEM_PROMPT = """
You're being asked to look up information to answer specific questions relating
to agentic memory.

Use your search tool to retrieve the correct answer

Then respond with the answer to the question
"""


async def memory_agent(question: str,
                       tools=[],
                       system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> tuple[str, str]:
    agent = Agent(
        name="Memory Agent",
        instructions=system_prompt,
        tools=tools,
    )
    result = await Runner.run(agent, question)
    return question, result.final_output


async def search_all(search_tool, judgments, eval_fn, limit=10):
    task_metadatas = {}
    tasks = []
    for _, judgment in judgments.head(limit).iterrows():
        question = judgment['question']
        metadata = judgment.to_dict()
        task_metadatas[question] = metadata
        assert isinstance(question, str)
        task = asyncio.create_task(memory_agent(question, tools=[search_tool]))
        tasks.append(task)

    eval_inputs = []
    for completed in asyncio.as_completed(tasks):
        question, result = await completed
        metadata = task_metadatas[question]
        metadata['predicted_answer'] = result
        eval_inputs.append(metadata)
    results = eval_fn(eval_inputs)
    accuracy = sum(r['score'] for r in results) / len(results)
    print(accuracy)


def main():
    parser = argparse.ArgumentParser(description="Run memory agent evaluation.")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to evaluate")
    parser.add_argument("--dataset", choices=["amabench", "longmemevalv2"], required=True,
                        help="Dataset to use for evaluation")
    args = parser.parse_args()
    corpus, judgments, eval = load_dataset(args.dataset)

    indexed_column = None
    if args.dataset == "longmemevalv2":
        indexed_column = ("Agent Thought: " + corpus["thought"].fillna("")
                          + " Agent Action: " + corpus["action"].fillna("")
                          + " Agent Goal: " + corpus["goal"].fillna(""))
    elif args.dataset == "amabench":
        indexed_column = "Agent Observation: " + corpus["observation"].fillna("")

    embeddings = model.encode(indexed_column.to_numpy(),
                              show_progress_bar=True, convert_to_numpy=True)

    @function_tool
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

    asyncio.run(search_all(limit=args.limit,
                           search_tool=search_memories,
                           eval_fn=eval,
                           judgments=judgments))


if __name__ == "__main__":
    main()
