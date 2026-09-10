import pandas as pd
from pathlib import Path


def load_episodes(root_path: Path | str) -> pd.DataFrame:
    return pd.read_json(Path(root_path) / "test" / "open_end_qa_set.jsonl", lines=True)


def judgments(root_path: Path | str) -> pd.DataFrame:
    episodes = load_episodes(root_path)
    # qa_pairs = episodes['qa_pairs']

    judgments = []
    for _, episode in episodes.iterrows():
        episode_id = episode['episode_id']
        episode_qas = episode['qa_pairs']
        for qa in episode_qas:
            qa['episode_id'] = episode_id
            judgments.append(qa)
    return pd.DataFrame(judgments)
