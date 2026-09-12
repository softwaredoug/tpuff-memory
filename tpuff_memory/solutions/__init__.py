import pandas as pd
from agents import Agent


def build_agent(solution: str, corpus: pd.DataFrame,
                failure_hook) -> Agent:
    if solution == "naive":
        from .naive import build_agent
        return build_agent(corpus, failure_hook)
    elif solution == "naive_tpuff":
        from .naive_tpuff import build_agent
        return build_agent(corpus, failure_hook)
    elif solution == "phrase_tpuff":
        from .phrase_tpuff import build_agent
        return build_agent(corpus, failure_hook)
    elif solution == "naive_entity":
        from .naive_entities import build_agent
        return build_agent(corpus, failure_hook)
    raise ValueError(f"Unknown solution: {solution}")
