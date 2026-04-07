# AgenticRAGLab

A hands-on lab that picks up where the [EscalationLab](https://github.com/FrankLaVigne/EscalationLab) left off — building agentic control structures around RAG pipelines, governed by the same principle: **escalation of effort must be justified by evidence**.

Uses the [Basic Fantasy RPG](https://www.basicfantasy.org/) rulebook as a realistic test domain to demonstrate why passive retrieve-then-answer pipelines break, and how an agent loop fixes the architectural gap without changing the model or the data.

## Structure

| Section | Directory | Topic |
|---------|-----------|-------|
| 00 | `00_Setup/` | Orientation, environment setup, and workbench configuration |
| 01 | `01_WhyPassiveRAGBreaks/` | Analyze the 4 failures the Escalation Lab could not fix — classify them as irrelevant retrieval, implicit reasoning, or out-of-scope |
| 02 | `02_TheAgentLoop/` | Build the agent loop: retrieval evaluation, query rewriting, and a decide-before-answering control structure |
| 03 | `03_DefiningTools/` | Define three callable tools (retrieval, calculator, no-answer) and demonstrate that tool selection quality depends on description quality |
| 04 | `04_RunningTheAgentLoop/` | Wire tools to the model, trace a full reasoning chain, and run all 10 evaluation questions through the agent loop |
| 05 | `05_Evaluation/` | Score the agent loop on two dimensions — answer correctness and reasoning correctness — using a 2x2 reliability matrix |
| 06 | `06_Synthesis/` | Facilitated discussion: when is an agent loop justified, and where does it sit on the escalation ladder? |

## The Core Argument

The Escalation Lab produced a RAG pipeline that answers 6 of 10 evaluation questions correctly. Best-of-N sampling and fine-tuning did not fix the remaining 4. Those failures are not model problems. They are architecture problems.

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

- **IBM Granite 3.2 8B Instruct** — Language model for generation
- **Granite Embedding 30M English** — Local embedding model for semantic search
- **ChromaDB** — In-process vector store
- **OpenAI Python client** — Interface to MaaS endpoint
- **Red Hat MaaS** — Model serving infrastructure

## Project Layout

```
AgenticRAGLab/
├── README.md
├── config.py                      # Shared config — loads .env and exposes credentials
├── .gitignore
├── docs/                          # Source documents (Basic Fantasy RPG PDF)
├── utils/
│   └── check_environment.ipynb    # Environment verification notebook
├── prebuilt/                      # Pre-generated results for offline use
│   ├── README.md
│   ├── eval_results.json          # Baseline evaluation from the Escalation Lab
│   ├── tool_definitions.json      # Tool schemas generated in Section 3
│   └── agent_loop_results.json    # Agent loop results from Section 4
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
└── 06_Synthesis/                  # Section 06
    └── 06_Synthesis.ipynb
```

## Connection to the Escalation Lab

This lab is a direct continuation of the [EscalationLab](https://github.com/FrankLaVigne/EscalationLab). The evaluation artifact in `prebuilt/eval_results.json` was produced at the end of that lab's pipeline. If you completed the Escalation Lab and have your own results, you can point the notebooks at your generated files instead.

The Escalation Lab demonstrated that some failures survive every improvement applied within a passive architecture. This lab demonstrates that those failures are architectural, and introduces the control structure that resolves them.
