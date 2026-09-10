"""Eval code as taken from AMA Bench source."""
from typing import List, Dict, Any
from tqdm import tqdm
import re
from openai import OpenAI
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
openai = OpenAI(api_key=OPENAI_API_KEY)


def call_openai(prompt, model="gpt-5-mini", max_tokens=2048):
    reasoning_effort = "minimal"

    response = openai.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_tokens,
        reasoning={"effort": reasoning_effort},
    )
    return response.output_text


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Strip whitespace
    """
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    return normalize_text(text).split()


def compute_f1_score(predicted: str, golden: str) -> float:
    """
    Compute token-level F1 score.

    Returns:
        F1 score between 0.0 and 1.0
    """
    pred_tokens = tokenize(predicted)
    gold_tokens = tokenize(golden)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    # Count common tokens
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    common = pred_counter & gold_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_llm_as_judge(
    question: str,
    golden_answer: str,
    predicted_answer: str,
    task_description: str = "",
    task_type: str = "",
    episode_id: str = "",
) -> float:
    """
    Use LLM to judge answer quality with binary (yes/no) judgment.

    Args:
        question: The question asked
        golden_answer: Reference answer
        predicted_answer: Model's predicted answer
        task_description: Description of the task context (optional)
        task_type: Type of the task (optional)
        episode_id: Episode ID for reference (optional)

    Returns:
        Binary score: 1.0 if correct, 0.0 if incorrect
    """
    # Build context information
    context_parts = []
    if task_type:
        context_parts.append(f"Task Type: {task_type}")
    if episode_id:
        context_parts.append(f"Episode ID: {episode_id}")
    if task_description:
        context_parts.append(f"Task Context: {task_description}")

    context_str = "\n".join(context_parts) if context_parts else ""

    judge_prompt = f"""You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context_str}

Question: {question}

Reference Answer: {golden_answer}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:<think></think>"""

    try:
        # Increase max_tokens to handle potential thinking tags and ensure complete response
        response = call_openai(judge_prompt, max_tokens=2048)

        # Remove thinking tags if present (some models use <think>...</think>)
        response_cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response_cleaned = response_cleaned.strip()

        # Extract the last occurrence of yes or no (case-insensitive)
        # This handles cases where model explains before answering
        response_lower = response_cleaned.lower()

        # Find all matches of "yes" or "no" as complete words
        yes_matches = list(re.finditer(r'\byes\b', response_lower))
        no_matches = list(re.finditer(r'\bno\b', response_lower))

        # Determine which comes last
        last_yes_pos = yes_matches[-1].start() if yes_matches else -1
        last_no_pos = no_matches[-1].start() if no_matches else -1

        if last_yes_pos > last_no_pos:
            return 1.0
        elif last_no_pos > last_yes_pos:
            return 0.0
        else:
            # Fallback to F1 if no valid response
            print(f"Warning: Could not parse LLM judge response: '{response}'. Falling back to F1 score.")
            return compute_f1_score(predicted_answer, golden_answer)
    except Exception as e:
        print(f"Error: LLM judge failed: {e}")
        raise


def evaluate_batch(
    qa_results: List[Dict[str, Any]],
    max_workers: int = 3,
) -> List[Dict[str, Any]]:
    """
    Evaluate a batch of QA results using LLM-as-judge with concurrent execution.

    Args:
        qa_results: List of QA result dictionaries containing:
            - episode_id, question, predicted_answer, golden_answer, etc.
        judge_client: ModelClient for LLM judge
        max_workers: Maximum number of concurrent workers

    Returns:
        List of evaluated results with scores added
    """

    def evaluate_single(result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single QA result."""
        score = compute_llm_as_judge(
            question=result['question'],
            golden_answer=result['golden_answer'],
            predicted_answer=result['predicted_answer'],
            task_description=result.get('task_description', ''),
            task_type=result.get('task_type', ''),
            episode_id=str(result.get('episode_id', '')),
        )
        result['score'] = score
        return result

    # Use thread pool for concurrent evaluation
    evaluated_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_result = {
            executor.submit(evaluate_single, result): result
            for result in qa_results
        }

        with tqdm(total=len(qa_results), desc="Evaluating QA pairs", unit="pair") as pbar:
            for future in as_completed(future_to_result):
                evaluated_results.append(future.result())
                pbar.update(1)

    return evaluated_results
