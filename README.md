# Experiments with Turbopuffer in Agentic memory

How can we create easy to use, high scale agentic memory with Turbopuffer

## Solutions (n=100)

Solutions all with [LongMemEval-v2](https://github.com/xiaowu0162/LongMemEval-V2). They can be fairly non-deterministic. All
stats reported are accuracy.

## Naive search

Just flatten every agent step, embed a description with MiniLM, and search them

```
uv run python -m tpuff_memory.agent  --dataset longmemevalv2 --solution naive_tpuff --limit 100
```

Gives understandably poor accuracy, over very few tool calls
```
ACC=0.25 tool_calls(mean)=1.28
```

## Add phrase tool

Basically

- when you seen an entity, search for otherh context by phrase to learn about the entity
- return an evidence chain

Allow phrase search:

```
 uv run python -m tpuff_memory.agent  --dataset longmemevalv2 --solution phrase_tpuff --limit 100
```

Gives

```
ACC=0.48 tool_calls(mean)=2.02
```

### Download dataset

Download the dataset to longmemeval-v2 with huggingface command:

```
hf download xiaowu0162/longmemeval-v2 --repo-type dataset --local-dir ./longmemeval-v2 tpuff-memory$
```
