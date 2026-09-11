from .longmemeval_v2 import documents
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from collections import defaultdict
from platformdirs import user_cache_dir
from pathlib import Path
import json
import asyncio


class EntitiesMentioned(BaseModel):
    """A list of jargony or ambiguous entities mentioned in the provided text (vague references, proper nouns, unknown topics, etc)."""
    entity_names: list[str] = Field(..., description="Entities mentioned")


default_entities_system_prompt = """
In the text below there are entities, artifacts, proper nouns, identities, etc not part of your general knowledge or seem
to refer to something outside this text.

We want to track every mention of them.

Please list the unresolved entities you don't know about from this text

Only list the names of the entities, no fluff

Don't list pronouns or generic terms like "button", "text", "image", "link", etc.
Only list proper nouns, unique identifiers, or specific references that are not part of your general knowledge.
"""


ui_trace_system_prompt = """
Below is an accessibility_tree based view of a UI.

there are entities, artifacts, proper nouns, identities, etc not part of your general knowledge or seem
to refer to something outside this text.

We want to track every mention of them.

Please list the unresolved entities you don't know about from this text.

Only list the names of the entities, no fluff.

Don't list pronouns or generic terms like "button", "text", "image", "link", etc.
Only list proper nouns, unique identifiers, or specific references that are not part of your general knowledge.
"""


def load_cache(cache_name):
    cache_dir = Path(user_cache_dir(cache_name))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = f"{cache_dir}/entities.json"
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_cache(cache_name, cache):
    cache_dir = Path(user_cache_dir(cache_name))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = f"{cache_dir}/entities.json"
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def make_cache_key(model, system_prompt, input, text_format: BaseModel):
    json_schema = text_format.model_json_schema()
    as_str = json.dumps(json_schema, sort_keys=True)
    return f"{model}_{hash(system_prompt)}_{hash(input)}_{hash(as_str)}"


async def call_openai_cached(client: AsyncOpenAI, cache_name: str,
                             model: str, system_prompt: str, input: str, text_format):
    cache = load_cache(cache_name)
    cache_key = make_cache_key(model, system_prompt, input, text_format)
    if cache_key in cache:
        restored = text_format.model_validate_json(cache[cache_key])
        return restored
    response = await client.responses.parse(
        model=model,
        instructions=system_prompt,
        input=input,
        text_format=text_format
    )
    assert response.output_parsed is not None, "Failed to parse response into SearchQueries"
    cache[cache_key] = response.output_parsed.model_dump_json()
    save_cache(cache_name, cache)
    return response.output_parsed


async def extract_entities(client: AsyncOpenAI,
                           doc_id: str,
                           column: str,
                           text: str,
                           system_prompt=default_entities_system_prompt) -> tuple[EntitiesMentioned, str, str]:
    input = f"""Here's the text to examine:

     {text}
    """
    response = await call_openai_cached(client, cache_name="longmemevalv2_entities",
                                        model="gpt-5-mini",
                                        system_prompt=system_prompt,
                                        input=input,
                                        text_format=EntitiesMentioned)
    return response, column, doc_id


async def extract_all(corpus, limit=100):
    documents_df = corpus
    # shuffle
    # documents_df = documents_df.sample(frac=1).reset_index(drop=True).head(limit)
    documents_df = documents_df.head(limit)
    tasks = []
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    async with AsyncOpenAI() as client:
        all_goals = documents_df[['goal', 'doc_id']].drop_duplicates()

        async def process(client, column, doc_id, text, system_prompt=default_entities_system_prompt):
            """Prevent too many concurrent requests to OpenAI API."""
            async with semaphore:
                return await extract_entities(client, column, doc_id, text, system_prompt)

        for _, row in all_goals.iterrows():
            doc_id = row['doc_id']
            assert isinstance(doc_id, str), f"doc_id is not a string: {doc_id}"
            text = row['goal']
            assert isinstance(text, str), f"goal is not a string: {text}"
            task = asyncio.create_task(process(client, column='goal',
                                               doc_id=doc_id,
                                               text=text))
            tasks.append(task)
        for _, row in documents_df.iterrows():
            doc_id = row['doc_id']
            assert isinstance(doc_id, str), f"doc_id is not a string: {doc_id}"
            text = row['thought']
            assert isinstance(text, str), f"thought is not a string: {text}"
            task = asyncio.create_task(process(client, column='thought',
                                               doc_id=doc_id,
                                               text=text))
            tasks.append(task)
            accessibility_tree = row['accessibility_tree']
            assert isinstance(accessibility_tree, str), f"accessibility_tree is not a string: {accessibility_tree}"
            task = asyncio.create_task(process(client, column='accessibility_tree',
                                               doc_id=doc_id,
                                               text=accessibility_tree,
                                               system_prompt=ui_trace_system_prompt))
            tasks.append(task)

        entity_index = defaultdict(list)

        for completed in asyncio.as_completed(tasks):
            try:
                result = await completed
                resp, col, doc_id = result
                if isinstance(resp, EntitiesMentioned):
                    print(f"Row {doc_id}--{col} entities: {resp.entity_names}")
                    entity_index[doc_id].append(resp.entity_names)
            except Exception as e:
                print(f"Error processing row {e}")
                raise
        return entity_index


if __name__ == "__main__":
    corpus = documents('longmemeval-v2')
    asyncio.run(extract_all(corpus, limit=10))
