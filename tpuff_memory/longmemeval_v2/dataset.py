import pandas as pd
from pathlib import Path


def judgments(path: Path | str):
    qs = pd.read_json(Path(path) / "questions.jsonl", lines=True)
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
            flattened.append(step)
    return pd.DataFrame(flattened)
