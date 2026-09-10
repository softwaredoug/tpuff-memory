import pandas as pd
from pathlib import Path


def load_episodes(root_path: Path | str) -> pd.DataFrame:
    return pd.read_json(Path(root_path) / "test" / "open_end_qa_set.jsonl", lines=True)


def judgments(root_path: Path | str) -> pd.DataFrame:
    episodes = load_episodes(root_path)
    # qa_pairs = episodes['qa_pairs']

    judgments = []
    for _, episode in episodes.iterrows():
        episode_qas = episode['qa_pairs']
        for qa in episode_qas:
            qa['episode_id'] = episode['episode_id']
            qa['episode_task'] = episode['task']
            qa['golden_answer'] = qa['answer']
            qa['episode_task_type'] = episode['task_type']
            judgments.append(qa)
    return pd.DataFrame(judgments)


def documents(root_path: Path | str) -> pd.DataFrame:
    """Flattened, indexable documetns with pointers back to episode metadata."""
    episodes = load_episodes(root_path)

    all_trajectories = []
    for _, episode in episodes.iterrows():
        trajectories = episode['trajectory']
        for trajectory in trajectories:
            trajectory['episode_task'] = episode['task']
            trajectory['episode_task_type'] = episode['task_type']
            trajectory['dataset'] = 'amabench'

            all_trajectories.append(trajectory)

    all_trajectories = pd.DataFrame(all_trajectories)
    all_trajectories['observation'] = all_trajectories['observation'].fillna('', inplace=True)
    assert len(all_trajectories[all_trajectories['observation'].isna()]) == 0
    return all_trajectories
