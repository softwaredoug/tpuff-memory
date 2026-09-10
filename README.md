## Experiments with Turbopuffer in Agentic memory

How can we create easy to use, high scale agentic memory with Turbopuffer

### Naive solutions

Just flatten every agent step, embed them with MiniLM, and search them

```
uv run python -m tpuff_memory.agent  --dataset longmemevalv2 --solution naive_tpuff --limit 10
```

Gives understandably poor results
```
0.1
```
