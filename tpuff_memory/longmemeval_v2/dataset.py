import pandas as pd
from pathlib import Path
import json
from typing import Literal
import os


# Only retrieve questions + relevant questions for override_question_id
override_question_id = os.environ.get("OVERRIDE_QUESTION_ID", None)


def judgments(path: Path | str):
    qs = pd.read_json(Path(path) / "questions.jsonl", lines=True)
    qs['golden_answer'] = qs['answer']
    if override_question_id is not None:
        print(f"Filtering to only question id {override_question_id}")
        qs = qs[qs['id'] == override_question_id]
    return qs


def _maybe_filter_to_haystacks(flattened: pd.DataFrame) -> pd.DataFrame:
    if override_question_id is not None:
        small_haystacks = haystacks('./longmemeval-v2', size='small')
        relevant_trajectories = small_haystacks[override_question_id]
        print(f"Filtering to only trajectories relevant to question id {override_question_id}: {relevant_trajectories}")
        flattened = flattened.loc[flattened['trajectory_id'].isin(relevant_trajectories), :]
        print(f"Filtered flattened trajectories to {len(flattened)} rows")
        return flattened
    return flattened


def documents(path: Path | str) -> pd.DataFrame:
    """Flattened step-wise view of every chat interaction."""
    trajectories = pd.read_json(Path(path) / "trajectories.jsonl", lines=True)
    flattened = []
    for _, trajectory in trajectories.iterrows():
        steps = trajectory['states']
        for step in steps:
            step['trajectory_id'] = trajectory['id']
            step['domain'] = trajectory['domain']
            step['goal'] = trajectory['goal']
            step['environment'] = trajectory['environment']
            step['outcome'] = trajectory['outcome']
            step['start_url'] = trajectory['start_url']
            step['dataset'] = 'longmemevalv2'
            step['doc_id'] = f"{trajectory['id']}_{step['step']}"
            if step['thought'] is None or isinstance(step['thought'], float):
                step['thought'] = ''
            flattened.append(step)
    return _maybe_filter_to_haystacks(pd.DataFrame(flattened))


def haystacks(path: Path | str, size: Literal['small', 'medium']) -> dict:
    path = Path(path)
    if size in ('small', 'medium'):
        path = path / "haystacks" / f"lme_v2_{size}.json"
        return json.load(open(path, 'r'))
    else:
        raise ValueError(f"Unknown haystack type: {type}")
