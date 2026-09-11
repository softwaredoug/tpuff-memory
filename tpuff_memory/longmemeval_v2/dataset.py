import pandas as pd
from pathlib import Path
import json
from typing import Literal


def judgments(path: Path | str):
    qs = pd.read_json(Path(path) / "questions.jsonl", lines=True)
    qs['golden_answer'] = qs['answer']
    return qs


def documents(path: Path | str):
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
    return pd.DataFrame(flattened)


def haystacks(path: Path | str, size: Literal['small', 'medium']) -> dict:
    path = Path(path)
    if size in ('small', 'medium'):
        path = path / "haystacks" / f"lme_v2_{size}.json"
        return json.load(open(path, 'r'))
    else:
        raise ValueError(f"Unknown haystack type: {type}")
