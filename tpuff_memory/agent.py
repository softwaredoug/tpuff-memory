import asyncio
from agents import Agent, Runner
from agents.run_context import RunContextWrapper
from .dataset import load_dataset
from .solutions import build_agent
import argparse
import pandas as pd
from .context import RequestContext


async def run_agent(agent: Agent,
                    question_id: int,
                    question: str):
    context = RequestContext(question=question,
                             question_id=question_id)
    result = await Runner.run(agent,
                              question,
                              context=context)
    return question, result.final_output


def report_tool_failure(
    context: RunContextWrapper[RequestContext],
    error: Exception,
) -> str:
    print(f"‼️ Memory search failed for question {context.context.question[:20]}... (ID: {context.context.question_id}) with error: {error}")
    return "The memory search backend failed. Continue without search results."


async def search_all(agent, judgments, eval_fn, limit=10):
    task_metadatas = {}
    tasks = []
    for idx, judgment in judgments.head(limit).iterrows():
        question = judgment['question']
        metadata = judgment.to_dict()
        task_metadatas[question] = metadata
        assert isinstance(question, str)
        task = asyncio.create_task(run_agent(agent,
                                             question=question,
                                             question_id=idx))
        tasks.append(task)

    eval_inputs = []
    for completed in asyncio.as_completed(tasks):
        question, result = await completed
        metadata = task_metadatas[question]
        metadata['predicted_answer'] = result
        eval_inputs.append(metadata)
    results = pd.DataFrame(eval_fn(eval_inputs))
    import pdb; pdb.set_trace()
    accuracy = results['score'].sum() / len(results)
    print(accuracy)


def main():
    parser = argparse.ArgumentParser(description="Run memory agent evaluation.")
    parser.add_argument("--limit", type=int, default=10, help="Number of questions to evaluate")
    parser.add_argument("--dataset", choices=["amabench", "longmemevalv2"], required=True,
                        help="Dataset to use for evaluation")
    parser.add_argument("--solution", choices=["naive", "naive_tpuff", "naive_entity"], required=True, help="Solution to evaluate")
    args = parser.parse_args()
    corpus, judgments, eval = load_dataset(args.dataset)
    agent = build_agent(args.solution, corpus,
                        failure_hook=report_tool_failure)

    asyncio.run(search_all(agent,
                           limit=args.limit,
                           eval_fn=eval,
                           judgments=judgments))


if __name__ == "__main__":
    main()
