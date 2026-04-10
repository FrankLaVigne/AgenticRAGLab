# Pre-Built Outputs

This directory contains pre-generated results so participants can follow along without live execution. Each notebook includes a `USE_PREBUILT = False` fallback cell that loads from these files.

| File | Source | Description |
|------|--------|-------------|
| `eval_results.json` | Escalation Lab | Final evaluation results from the Escalation Lab pipeline: 10 questions scored against the passive RAG baseline |
| `bfrpg_chunks.json` | Basic Fantasy RPG | Pre-chunked corpus passages (30 chunks) used by `retriever.py` for lightweight TF-IDF retrieval |
| `tool_definitions.json` | Section 3 | OpenAI-compatible tool schemas for rag_retrieval, calculator, and no_answer |
| `agent_loop_results.json` | Section 4 | Agent loop results for all 10 questions, including full execution traces |
