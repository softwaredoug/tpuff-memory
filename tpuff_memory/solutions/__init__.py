import pandas as pd
from agents import Agent


def build_agent(solution: str, corpus: pd.DataFrame) -> Agent:
    if solution == "naive":
        from .naive import build_agent
        return build_agent(corpus)
    elif solution == "naive_tpuff":
        from .naive_tpuff import build_agent
        return build_agent(corpus)
    raise ValueError(f"Unknown solution: {solution}")
