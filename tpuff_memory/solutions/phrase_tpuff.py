import os
import turbopuffer
import pandas as pd
from agents import Agent, function_tool
from agents.tool_context import ToolContext
from ..context import RequestContext
from sentence_transformers import SentenceTransformer
import tqdm
from pydantic import BaseModel, Field


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
            "artifact": row['artifact'],
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

    def index_docs(self, corpus, force=False):
        """
        Index the documents into TurboPuffer.
        """
        ns_name = "agentmem-" + corpus['dataset'].iloc[0]
        expected_count = len(corpus)
        self.ns = self.tpuf.namespace(ns_name)
        if not ns_exists(self.tpuf, ns_name, expected_count) or force:
            embeddings = model.encode(corpus['description'].to_numpy(), show_progress_bar=True, convert_to_numpy=True)
            with tqdm.tqdm(total=expected_count, desc="Indexing documents") as pbar:
                for batch in tpuff_batch(corpus, embeddings):
                    self.ns.write(
                        upsert_rows=batch,
                        distance_metric="cosine_distance",
                        schema={
                            "content": {
                                "type": "string",
                                "full_text_search": {
                                    "tokenizer": "word_v4",
                                    "language": "english",
                                    "stemming": True,
                                    "remove_stopwords": False,
                                    "case_sensitive": False
                                },
                                "filterable": False
                            },
                            "artifact": {
                                "type": "string",
                                "full_text_search": {
                                    "tokenizer": "word_v4",
                                    "language": "english",
                                    "stemming": True,
                                    "remove_stopwords": False,
                                    "case_sensitive": False
                                },
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

    def _format_results(self, tpuff_results):
        results: list[dict] = []
        for row in tpuff_results.rows or []:
            content = row['content']
            result = {
                "id": row['id'],
                "score": row['$dist'],
                "content": content if isinstance(content, str) else ""
            }
            results.append(result)
        return results

    def query(self, query, top_k=5) -> list[dict]:
        print(f"Querying TurboPuffer for top {top_k} results related to: {query}")
        if self.ns is None:
            raise RuntimeError("Namespace is not initialized. Please index documents first.")
        query_embedding = model.encode([query])[0].tolist()
        ns_results = self.ns.query(
            rank_by=("vector", "ANN", query_embedding),
            top_k=top_k,
            include_attributes=["content"],
        )
        return self._format_results(ns_results)

    def context_mentioning_terms(self, terms: list[str], top_k=5) -> dict:
        """
        Query TurboPuffer for context mentioning the given terms.
        """
        print(f"Querying TurboPuffer for top {top_k} results mentioning terms: {terms}")
        if self.ns is None:
            raise RuntimeError("Namespace is not initialized. Please index documents first.")
        results = {}
        for term in terms:
            ns_results = self.ns.query(
                rank_by=(
                    "Sum",
                    (
                        ("content", "BM25", term),
                        ("artifact", "BM25", term),
                    )
                ),
                top_k=top_k,
                include_attributes=["content"],
                filters=(
                    "Or", (
                        ("content", "ContainsTokenSequence", term),
                        ("artifact", "ContainsTokenSequence", term),
                    ),
                )
            )
            term_results = self._format_results(ns_results)
            results[term] = term_results
        return results


class RetrievalEvidence(BaseModel):
    """A single item in the evidence chain and why its useful. While you may have retrieved many chat turns, this only details whats relevant"""

    doc_id: str = Field(..., description="The document ID of the useful evidence.")

    relevance_to_question: str = Field(..., description="Why this evidence is relevant to the question being asked.")


class Answer(BaseModel):
    evidence_chain: list[RetrievalEvidence] = Field(..., description="The chain of evidence used to arrive at the answer.")

    final_output: str = Field(..., description="The final answer to the question being asked. Possibly boxed.")

    def __str__(self):
        evidence_str = "\n".join(
            [f"Doc ID: {e.doc_id}, Relevance: {e.relevance_to_question}" for e in self.evidence_chain])
        return f"Evidence Chain:\n{evidence_str}\n\nFinal Answer:\n{self.final_output}"


DEFAULT_SYSTEM_PROMPT = """
In the past, an AI agent laboriously executed different tasks. And that knowledge was hard-earned
and we want to reuse it. The knowledge isn't part of your training set, its part of one companies
specific processes, apps, knowledge base, etc

You'll being asked about specific tasks or work artifacts in this dataset to save time and effort
having to "relearn" the same knowledge.

To do this:
- Query the past traces with the tools given
- Understand the structure of the dataset, what goals the agent/user was trying to achieve, and what actions were taken
- Build an evidence chain and show your work getting to the right answer
- Finally conclude with the correct answer

Don't assume. Don't hallucinate. Seek evidence.

You have tools to help explore evidence:
    - search_memories: will search through past memories
    - define_terms: given a phrase, like unknown jargon, searches for that exact phrase. IE if you want to look
      up any unknown terminology referenced, like "Quarterly Earnings Report" you could search for that exact
      phrase

"""

# 1d56a4d6, Step 11


def build_agent(corpus: pd.DataFrame,
                failure_hook) -> Agent:
    indexed_column = None
    artifact = None
    dataset = corpus['dataset'].iloc[0]
    if dataset == "longmemevalv2":
        indexed_column = ("Agent Thought: " + corpus["thought"].fillna("")
                          + " Agent Action: " + corpus["action"].fillna("")
                          + " Agent Goal: " + corpus["goal"].fillna(""))
        artifact = corpus['accessibility_tree']
    elif dataset == "amabench":
        indexed_column = "Agent Observation: " + corpus["observation"].fillna("")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    corpus['description'] = indexed_column
    corpus['artifact'] = artifact if artifact is not None else ""

    tpuff_index = TurboPufferIndex()
    tpuff_index.index_docs(corpus, force=False)

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
        results: list[dict] = []
        for result in tpuff_results:
            print(f"{q_id} -- Result ID: {result['id']}, Score: {result['score']}")
            if not corpus[corpus['doc_id'] == result['id']].empty:
                results.append(corpus[corpus['doc_id'] == result['id']].iloc[0].to_dict())
            else:
                pass  # This happens when debugging a smaller haystack thats smaller than the tpuff corpus

        return results

    @function_tool(failure_error_function=failure_hook,
                   timeout=60)
    async def find_with_phrases(ctx: ToolContext[RequestContext],
                                phrases: list[str]) -> dict:
        """Find historical context with the exact phrase mentioned

        Params: terms: list of phrases to lookup in the corpus

        Lookup context for unknown terminology: jargon, phrases, identifiers."""
        req_context: RequestContext = ctx.context
        q_id = req_context.question_id
        print(f"{q_id} --Defining terms: {phrases}")
        results = {}
        all_term_results = tpuff_index.context_mentioning_terms(phrases, top_k=5)
        for term, term_results in all_term_results.items():
            hydrated_term_results = []
            for term_result in term_results:
                print(f"{q_id},{term} -- Result ID: {term_result['id']}")
                if not corpus[corpus['doc_id'] == term_result['id']].empty:
                    hydrated_term_results.append(corpus[corpus['doc_id'] == term_result['id']].iloc[0].to_dict())
                else:
                    pass  # This happens when debugging a smaller haystack thats smaller than the tpuff corpus
            results[term] = hydrated_term_results
        return results

    agent = Agent(
        name="Memory Agent",
        instructions=DEFAULT_SYSTEM_PROMPT,
        tools=[search_memories, find_with_phrases],
        output_type=Answer
    )
    return agent
