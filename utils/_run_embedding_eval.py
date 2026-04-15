"""
One-shot script: run the same agent loop (verbatim from section 04) against all
10 eval questions using the Chroma retriever, save results to
prebuilt/agent_loop_embeddings_results.json as the graceful-degrade fallback for
section 07.

This is not part of the lab curriculum; it is a build-time utility. The bonus
notebook has equivalent cells learners run interactively.
"""
import ast
import json
import operator
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from openai import OpenAI
from utils.retriever_chroma import ChromaRetriever

collection = ChromaRetriever.load()
client = OpenAI(api_key=config.API_KEY, base_url=config.ENDPOINT_BASE)

with open(ROOT / "prebuilt" / "tool_definitions.json") as f:
    tool_definitions = json.load(f)["tool_definitions"]
with open(ROOT / "prebuilt" / "eval_results.json") as f:
    eval_data = json.load(f)

# ---- verbatim from 04_Running_the_Agent_Loop.ipynb cell 8 ---------------------

def rag_retrieval(query: str) -> dict:
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "distances"],
    )
    chunks = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        chunks.append({"text": doc, "distance": round(dist, 4)})
    return {"tool": "rag_retrieval", "query": query, "chunks": chunks}


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}

def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    raise ValueError(f"Unsupported operation: {ast.dump(node)}")


def calculator(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        return {"tool": "calculator", "expression": expression, "result": _safe_eval_node(tree)}
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
        return {"tool": "calculator", "expression": expression, "error": str(e)}


def no_answer() -> dict:
    return {
        "answer": "I do not have enough information to answer this question from the available corpus.",
        "tool": "no_answer",
    }


TOOL_DISPATCH = {"rag_retrieval": rag_retrieval, "calculator": calculator, "no_answer": no_answer}


def execute_tool(name, arguments):
    if name not in TOOL_DISPATCH:
        return json.dumps({"error": f"Unknown tool: {name}"})
    return json.dumps(TOOL_DISPATCH[name](**arguments), default=str)


# ---- verbatim from 04_Running_the_Agent_Loop.ipynb cell 10 --------------------

def run_agent_loop(question, tools, client, model_id, max_iterations=3, verbose=False):
    system_prompt = (
        "You are a rules assistant for Basic Fantasy RPG. "
        "Use the available tools to answer questions about the game rules. "
        "If the retrieved context is sufficient, answer the question directly. "
        "If the context is insufficient or irrelevant, use the no_answer tool. "
        "If the question requires a calculation, use the calculator tool. "
        "Always base your answer on tool results, not prior knowledge."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    trace = {"question": question, "steps": [], "final_answer": None, "iterations": 0}

    for iteration in range(1, max_iterations + 1):
        trace["iterations"] = iteration
        response = client.chat.completions.create(
            model=model_id, messages=messages, tools=tools, temperature=0.0
        )
        choice = response.choices[0]
        if choice.message.tool_calls:
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                result = execute_tool(name, args)
                trace["steps"].append({
                    "iteration": iteration, "tool": name, "arguments": args,
                    "result": json.loads(result),
                })
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            trace["final_answer"] = choice.message.content
            return trace

    messages.append({
        "role": "user",
        "content": "Based on the tool results above, provide your final answer to the original question.",
    })
    response = client.chat.completions.create(model=model_id, messages=messages, temperature=0.0)
    trace["final_answer"] = response.choices[0].message.content
    return trace


JUDGE_PROMPT = """You are an evaluation judge. Compare the EXPECTED answer to the ACTUAL answer.

The ACTUAL answer is correct if it conveys the same key facts as the EXPECTED answer,
even if the wording differs. Minor omissions of non-essential details are acceptable.

Respond with EXACTLY one JSON object:
{"classification": "pass" or "fail", "reason": "<one sentence>"}
"""


def judge(q, expected, actual):
    response = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": f"QUESTION: {q}\n\nEXPECTED: {expected}\n\nACTUAL: {actual}"},
        ],
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"classification": "error", "reason": "Could not parse judge response"}


results = eval_data["results"]
agent_loop_results = []

for r in results:
    print(f"[{r['id']}] {r['question'][:80]}", flush=True)
    trace = run_agent_loop(
        question=r["question"], tools=tool_definitions,
        client=client, model_id=config.MODEL_ID, max_iterations=3, verbose=False,
    )
    judgment = judge(r["question"], r["expected"], trace["final_answer"])
    entry = {
        "id": r["id"], "question": r["question"], "expected": r["expected"],
        "category": r["category"], "passive_classification": r["classification"],
        "agent_answer": trace["final_answer"], "iterations": trace["iterations"],
        "tools_used": [s["tool"] for s in trace["steps"]],
        "trace": trace["steps"],
        "agent_classification": judgment["classification"],
        "judge_reason": judgment.get("reason", ""),
    }
    agent_loop_results.append(entry)
    print(f"   -> {entry['agent_classification']}  iters={entry['iterations']}  tools={entry['tools_used']}",
          flush=True)

output = {
    "metadata": {
        "generated_by": "07_Bonus_AgentLoopWithEmbeddings (build-time run)",
        "model": config.MODEL_ID,
        "retriever": "chromadb + ibm-granite/granite-embedding-30m-english",
        "max_iterations": 3,
        "tools": [td["function"]["name"] for td in tool_definitions],
        "total_questions": len(agent_loop_results),
        "agent_passes": sum(1 for ar in agent_loop_results if ar["agent_classification"] == "pass"),
    },
    "results": agent_loop_results,
}

out_path = ROOT / "prebuilt" / "agent_loop_embeddings_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {out_path}")
print(f"Passes: {output['metadata']['agent_passes']}/10")
