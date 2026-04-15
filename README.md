# AgenticRAGLab

A hands-on lab that picks up where the [EscalationLab](https://github.com/FrankLaVigne/EscalationLab) left off, building agentic control structures around RAG pipelines, governed by the same principle: **escalation of effort must be justified by evidence**.

Uses the [Basic Fantasy RPG](https://www.basicfantasy.org/) rulebook as a realistic test domain to demonstrate why passive retrieve-then-answer pipelines break, and how an agent loop fixes the architectural gap without changing the model or the data.

## Structure

| Section | Directory | Topic |
|---------|-----------|-------|
| 00 | `00_Setup/` | Orientation, environment setup, and workbench configuration |
| 01 | `01_WhyPassiveRAGBreaks/` | Analyze the 4 failures the Escalation Lab could not fix; classify them as irrelevant retrieval, implicit reasoning, or out-of-scope |
| 02 | `02_TheAgentLoop/` | Build the agent loop: retrieval evaluation, query rewriting, and a decide-before-answering control structure |
| 03 | `03_DefiningTools/` | Define three callable tools (retrieval, calculator, no-answer) and demonstrate that tool selection quality depends on description quality |
| 04 | `04_RunningTheAgentLoop/` | Wire tools to the model, trace a full reasoning chain, and run all 10 evaluation questions through the agent loop |
| 05 | `05_Evaluation/` | Score the agent loop on two dimensions (answer correctness and reasoning correctness) using a 2x2 reliability matrix |
| 06 | `06_Synthesis/` | Facilitated discussion: when is an agent loop justified, and where does it sit on the escalation ladder? |
| 07 | `07_Bonus_AgentLoopWithEmbeddings/` | **Bonus:** Swap the TF-IDF retriever for ChromaDB + Granite embeddings, re-run the same loop, and check whether the architectural pattern generalizes |

## The Core Argument

The Escalation Lab produced a RAG pipeline that answers 8 of 10 evaluation questions correctly. Best-of-N sampling and fine-tuning did not fix the remaining 2. Those failures are not model problems. They are architecture problems.

A passive pipeline always retrieves, always answers, and always moves on. It has no mechanism to:

- Judge whether retrieval was sufficient before answering
- Rewrite a query and try again when retrieval is weak
- Abstain when the corpus does not contain the answer

The agent loop adds those three capabilities. The model stays the same. The retriever stays the same. The control structure changes. That is the difference between passive and agentic RAG.

```
Passive:    Question → Retrieve → Answer (always, in that order)

Agentic:    Question → Retrieve → Evaluate → Decide
                                     ↓
                              sufficient? → Answer
                              partial?    → Rewrite query → Retrieve again
                              irrelevant? → Abstain
```

## What This Lab Does Not Cover

The agent loop handles the abstain case correctly: when the corpus does not contain the answer, the system says so rather than hallucinating. But when the agent does answer, it does not surface which documents grounded that answer. In enterprise contexts, the ability to trace a generated response back to a specific source document is a compliance and governance asset, not just a debugging aid. Source provenance turns a black-box answer into an auditable one. A future extension of this lab would attach citations to the answer path, linking each claim to the chunk that supports it.

The lab classifies retrieval failures as irrelevant retrieval, implicit reasoning, or out-of-scope, but it does not distinguish between a query that was poorly formed and a corpus that is incomplete or poorly chunked. Both produce weak retrieval, but they require different fixes. The agent loop addresses the query side through evaluation and rewriting. Corpus quality (whether the right information was ingested, whether chunks preserve the structure needed to answer multi-fact questions, whether tables survived the chunking process intact) is a separate concern that sits upstream of the agent loop and outside the scope of this lab.

This lab intentionally keeps the model and retriever fixed and changes only the control structure. That is the right scope for demonstrating the agentic pattern in isolation. However, the production end-state for most enterprise deployments combines RAG with a fine-tuned model: fine-tuning handles deep domain fluency and consistent formatting, RAG handles real-time context and factual grounding. Neither alone is sufficient. This lab is one half of that architecture. The Escalation Lab covers the other half.

## Getting Started

### Prerequisites

- Python 3.12+
- JupyterLab
- API key and endpoint for a MaaS (Model as a Service) instance running Granite models
- Completed [EscalationLab](https://github.com/FrankLaVigne/EscalationLab) (recommended), or use the pre-built outputs in `prebuilt/`

### Setup

1. Set environment variables for the MaaS endpoint:

   ```
   MAAS_API_KEY=your-api-key
   MAAS_BASE_URL=https://your-maas-endpoint/v1
   MAAS_MODEL_ID=granite-3-2-8b-instruct
   ```

2. Install dependencies within the notebooks as needed (each notebook installs its own requirements).

3. Work through the notebooks in order, starting with `00_Setup/00_Setup_and_Orientation.ipynb`.

4. If you did not complete the Escalation Lab, each notebook includes a pre-built fallback cell that loads saved outputs so you can follow along without live execution.

## Key Technologies

- **IBM Granite 3.2 8B Instruct**: Language model for generation
- **OpenAI Python client**: Interface to MaaS endpoint
- **Red Hat MaaS**: Model serving infrastructure
- **Pure-Python TF-IDF retriever**: Lightweight retrieval over pre-chunked JSON (see note below)

### Why no ChromaDB or embedding model?

The original design called for ChromaDB with a local embedding model. We replaced it with a zero-dependency TF-IDF retriever backed by `prebuilt/bfrpg_chunks.json` for three reasons:

1. **Workshop portability**: ChromaDB and sentence-transformers add heavy native dependencies that complicate setup on constrained environments (containers, shared JupyterHub instances).
2. **The lab's point is the control structure, not the retriever.** Sections 1–2 demonstrate that the *architecture* around retrieval matters more than the retrieval engine itself. A simple retriever makes that argument more clearly.
3. **Reproducibility**: Pre-chunked JSON is deterministic and version-controllable. No embedding drift, no index rebuild step.

The retriever module (`retriever.py`) exposes the same `.query()` interface as ChromaDB, so swapping in a real vector store later requires changing only the import. Section 07 (bonus) does exactly that. It re-runs the full agent loop with a ChromaDB + Granite-embedding retriever and compares results, so you can see the pattern generalize without taking it on faith.

## Project Layout

```
AgenticRAGLab/
├── README.md
├── config.py                      # Shared config: loads .env and exposes credentials
├── .gitignore
├── docs/                          # Source documents (Basic Fantasy RPG PDF)
├── utils/
│   ├── check_environment.ipynb    # Environment verification notebook
│   ├── retriever.py               # Pure-Python TF-IDF retriever (replaces ChromaDB)
│   └── retriever_chroma.py        # Chroma + embedding retriever used by Section 7
├── prebuilt/                      # Pre-generated results for offline use
│   ├── README.md
│   ├── eval_results.json          # Baseline evaluation from the Escalation Lab
│   ├── bfrpg_chunks.json          # Pre-chunked corpus passages for retrieval
│   ├── tool_definitions.json      # Tool schemas generated in Section 3
│   ├── agent_loop_results.json    # Agent loop results from Section 4 (TF-IDF)
│   ├── agent_loop_embeddings_results.json  # Bonus run from Section 7 (embeddings)
│   └── chroma_agentic/            # Persisted Chroma collection for Section 7
├── 00_Setup/                      # Section 00
│   └── 00_Setup_and_Orientation.ipynb
├── 01_WhyPassiveRAGBreaks/        # Section 01
│   └── 01_Why_Passive_RAG_Breaks.ipynb
├── 02_TheAgentLoop/               # Section 02
│   └── 02_The_Agent_Loop.ipynb
├── 03_DefiningTools/              # Section 03
│   └── 03_Defining_Tools.ipynb
├── 04_RunningTheAgentLoop/        # Section 04
│   └── 04_Running_the_Agent_Loop.ipynb
├── 05_Evaluation/                 # Section 05
│   └── 05_Evaluation.ipynb
├── 06_Synthesis/                  # Section 06
│   └── 06_Synthesis.ipynb
├── 07_Bonus_AgentLoopWithEmbeddings/  # Section 07 (bonus)
│   └── 07_Bonus_Agent_Loop_with_Embeddings.ipynb
└── extras/
    └── WhatsNext.ipynb            # Post-lab guide: production concerns, further reading
```

## Connection to the Escalation Lab

This lab is a direct continuation of the [EscalationLab](https://github.com/FrankLaVigne/EscalationLab). The evaluation artifact in `prebuilt/eval_results.json` was produced at the end of that lab's pipeline. If you completed the Escalation Lab and have your own results, you can point the notebooks at your generated files instead.

The Escalation Lab demonstrated that some failures survive every improvement applied within a passive architecture. This lab demonstrates that those failures are architectural, and introduces the control structure that resolves them.
