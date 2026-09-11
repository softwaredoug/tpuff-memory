import os
import turbopuffer
import pandas as pd
from agents import Agent, function_tool
from agents.tool_context import ToolContext
from ..context import RequestContext
from sentence_transformers import SentenceTransformer
import tqdm


TPUF_API_KEY = os.getenv("TPUF_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')


def tpuff_batch(corpus, embeddings, batch_size=1024):
    """
    Turbopuffer rows batched
    """
    curr_batch = []
    for idx, row in corpus.iterrows():
        doc = {
            "id": row['doc_id'],
            "content": row['description'],
            "vector": embeddings[idx].tolist()
        }
        curr_batch.append(doc)
        if len(curr_batch) >= batch_size:
            yield curr_batch
            curr_batch = []
    yield curr_batch


def ns_exists(tpuf, ns_name, expected_count):
    ns = tpuf.namespace(ns_name)
    try:
        metadata = ns.metadata()
        if metadata.approx_row_count < expected_count:
            print(f"Namespace {ns_name} exists but has fewer rows ({metadata.approx_row_count}) than expected ({expected_count}).")
            return False
    except turbopuffer.NotFoundError:
        return False
    return True


class TurboPufferIndex:

    def __init__(self):
        self.tpuf = turbopuffer.Turbopuffer(
            api_key=TPUF_API_KEY,
            region="gcp-us-central1"
        )
        self.ns = None

    def index_docs(self, corpus):
        """
        Index the documents into TurboPuffer.
        """
        ns_name = "agentmem-" + corpus['dataset'].iloc[0]
        expected_count = len(corpus)
        self.ns = self.tpuf.namespace(ns_name)
        if not ns_exists(self.tpuf, ns_name, expected_count):
            embeddings = model.encode(corpus['description'].to_numpy(), show_progress_bar=True, convert_to_numpy=True)
            with tqdm.tqdm(total=expected_count, desc="Indexing documents") as pbar:
                for batch in tpuff_batch(corpus, embeddings):
                    self.ns.write(
                        upsert_rows=batch,
                        distance_metric="cosine_distance",
                        schema={
                            "content": {
                                "type": "string",
                                "full_text_search": True,
                                "filterable": False
                            }
                        }
                    )
                    pbar.update(len(batch))

            result = self.ns.query(
                rank_by=("id", "asc"),
                limit=1,
            )
            count = result.performance.approx_namespace_size
            print(f"Indexed {count} documents into TurboPuffer.")

    def query(self, query, top_k=5) -> list[dict]:
        print(f"Querying TurboPuffer for top {top_k} results related to: {query}")
        if self.ns is None:
            raise RuntimeError("Namespace is not initialized. Please index documents first.")
        query_embedding = model.encode([query])[0].tolist()
        print("Encoded, calling")
        ns_results = self.ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=top_k,
            include_attributes=["content"],
        )
        print("Results retrieved")
        results: list[dict] = []
        for row in ns_results.rows or []:
            content = row['content']
            result = {
                "id": row['id'],
                "score": row['$dist'],
                "content": content if isinstance(content, str) else ""
            }
            results.append(result)
        return results


DEFAULT_SYSTEM_PROMPT = """
You're being asked to look up information to answer specific questions relating
to agentic memory.

Use your search tool to retrieve the correct answer

Then respond with the answer to the question
"""


def build_agent(corpus: pd.DataFrame,
                failure_hook) -> Agent:
    indexed_column = None
    dataset = corpus['dataset'].iloc[0]
    if dataset == "longmemevalv2":
        indexed_column = ("Agent Thought: " + corpus["thought"].fillna("")
                          + " Agent Action: " + corpus["action"].fillna("")
                          + " Agent Goal: " + corpus["goal"].fillna(""))
    elif dataset == "amabench":
        indexed_column = "Agent Observation: " + corpus["observation"].fillna("")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    corpus['description'] = indexed_column

    tpuff_index = TurboPufferIndex()
    tpuff_index.index_docs(corpus)

    @function_tool(failure_error_function=failure_hook,
                   timeout=60)
    async def search_memories(ctx: ToolContext[RequestContext],
                              query: str):
        """Return 5 similar agent events tto query to help answer the question."""
        req_context: RequestContext = ctx.context
        q_id = req_context.question_id
        print(f"{q_id} -- Question: {req_context.question}")
        print(f"{q_id} -- Using query: {query}")
        tpuff_results = tpuff_index.query(query, top_k=5)
        print("HERE")
        print(f"{q_id} -- Retrieved {len(tpuff_results)} results from TurboPuffer.")
        results: list[dict] = []
        for result in tpuff_results:
            print(f"{q_id} -- Result ID: {result['id']}, Score: {result['score']}")
            results.append(corpus[corpus['doc_id'] == result['id']].iloc[0].to_dict())

        return results

    agent = Agent(
        name="Memory Agent",
        instructions=DEFAULT_SYSTEM_PROMPT,
        tools=[search_memories]
    )
    return agent
