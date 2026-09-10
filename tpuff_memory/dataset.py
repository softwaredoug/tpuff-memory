"""Load a dataset's functions given its name."""
from typing import Callable, Tuple
import pandas as pd


def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame, Callable]:
    if name == "amabench":
        path = "AMA-bench"
        from .ama_bench import documents, judgments, eval
        return documents(path), judgments(path), eval
    elif name == "longmemevalv2":
        from .longmemeval_v2 import documents, judgments, eval
        path = "longmemeval-v2"
        return documents(path), judgments(path), eval
    raise ValueError(f"Unknown dataset name: {name}")
