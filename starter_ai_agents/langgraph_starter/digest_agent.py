"""
Daily Digest Agent — parallel research using the LangGraph Send fan-out pattern.

Flow:
  START → fan_out_node (Send x N) → researcher_node (parallel) → compile_digest_node → save_node → END
"""

import os
import operator
from datetime import date
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from pydantic import BaseModel

from tools import wikipedia_search, web_search

load_dotenv()

TOPICS = ["AI news", "Python releases", "space exploration"]


class ResearchSummary(BaseModel):
    title: str          # one-line answer
    bullets: list[str]  # 3 key facts
    confidence: str     # "high" | "medium" | "low"


class DigestState(TypedDict):
    topics: list[str]
    current_topic: str                              # set per-branch by Send
    summaries: Annotated[list, operator.add]        # fan-in: each branch appends its ResearchSummary
    report: str | None


llm = ChatLiteLLM(
    model="openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
    api_base=os.environ["NEBIUS_BASE_URL"],
    api_key=os.environ["NEBIUS_API_KEY"],
    max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
    temperature=float(os.getenv("TEMPERATURE", 0.3)),
)

RESEARCHER_TOOLS = [wikipedia_search, web_search]


def build_digest_graph() -> CompiledStateGraph:

    # ── Fan-out node ───────────────────────────────────────────────────────────
    # Returns one Send per topic; LangGraph runs all researcher_node calls in parallel.
    def fan_out_node(state: DigestState):
        return [Send("researcher_node", {**state, "current_topic": t}) for t in state["topics"]]

    # ── Researcher node (one instance per topic, all parallel) ─────────────────
    def researcher_node(state: DigestState):
        topic = state["current_topic"]
        llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)

        # Step 1: LLM picks tools
        response = llm_with_tools.invoke([
            {"role": "system", "content": "Use wikipedia_search or web_search to find the latest information on the topic. Call at least one tool."},
            {"role": "user", "content": f"Research: {topic}"},
        ])
        print(f"\n[researcher_node] topic={topic!r} tools={[tc['name'] for tc in response.tool_calls]}")

        # Step 2: Execute tool calls
        WIKI_ERRORS = ("No page found", "Your query is ambiguous", "An error occurred")
        findings = []
        for tc in response.tool_calls:
            name, args = tc["name"], tc["args"]
            if name == "wikipedia_search":
                result_text = wikipedia_search.invoke(args)
                if any(result_text.startswith(e) for e in WIKI_ERRORS):
                    result_text = web_search.invoke({"query": topic})
                findings.append(result_text)
            elif name == "web_search":
                findings.append(web_search.invoke(args))

        if not findings:
            findings.append(web_search.invoke({"query": topic}))

        # Step 3: Structured summary — stored as dict so MemorySaver can serialize it
        combined = "\n".join(findings)
        result: ResearchSummary = llm.with_structured_output(ResearchSummary).invoke([
            {"role": "system", "content": "Summarize with a title (one-line answer), bullets (3 key facts), and confidence (high/medium/low)."},
            {"role": "user", "content": combined},
        ])
        print(f"\n[researcher_node] {topic!r} → {result.title}")
        return {"summaries": [result.model_dump()]}

    # ── Compile digest node ────────────────────────────────────────────────────
    def compile_digest_node(state: DigestState):
        sections = []
        for s in state["summaries"]:
            bullets = "\n".join(f"- {b}" for b in s["bullets"])
            sections.append(f"## {s['title']}\n\n{bullets}\n\n**Confidence:** {s['confidence']}")
        report = "\n\n---\n\n".join(sections)
        print(f"\n[compile_digest_node]: {len(state['summaries'])} summaries compiled")
        return {"report": report}

    # ── Save node ──────────────────────────────────────────────────────────────
    def save_node(state: DigestState):
        filename = f"digest_{date.today().isoformat()}.md"
        with open(filename, "a") as f:
            f.write(state["report"] + "\n\n---\n\n")
        print(f"\n[save_node]: saved to {filename}")
        return {}

    graph = StateGraph(DigestState)
    graph.add_node("researcher_node", researcher_node)
    graph.add_node("compile_digest_node", compile_digest_node)
    graph.add_node("save_node", save_node)

    graph.add_conditional_edges(START, fan_out_node, ["researcher_node"])
    graph.add_edge("researcher_node", "compile_digest_node")
    graph.add_edge("compile_digest_node", "save_node")
    graph.add_edge("save_node", END)

    return graph.compile(interrupt_before=["save_node"], checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_digest_graph()
    config = {"configurable": {"thread_id": "digest-session-1"}}

    initial_state = DigestState(
        topics=TOPICS,
        current_topic="",
        summaries=[],
        report=None,
    )

    final_state = graph.invoke(initial_state, config=config)
    print("\n" + final_state["report"])

    save = input("\nSave digest? (y/n): ").strip().lower()
    if save in ("y", "yes"):
        graph.invoke(None, config=config)
        print("Digest saved.")
