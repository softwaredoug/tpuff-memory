import asyncio
from agents import Agent, Runner
from .dataset import load_dataset
from .solutions.naive import build_agent
import argparse


async def run_agent(agent: Agent,
                    question: str):
    result = await Runner.run(agent, question)
    return question, result.final_output


async def search_all(agent, judgments, eval_fn, limit=10):
    task_metadatas = {}
    tasks = []
    for _, judgment in judgments.head(limit).iterrows():
        question = judgment['question']
        metadata = judgment.to_dict()
        task_metadatas[question] = metadata
        assert isinstance(question, str)
        task = asyncio.create_task(run_agent(agent, question))
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
    agent = build_agent(corpus)

    asyncio.run(search_all(agent,
                           limit=args.limit,
                           eval_fn=eval,
                           judgments=judgments))


if __name__ == "__main__":
    main()
