"""
Build a research agent that can perform research tasks using the LangGraph framework. The agent should be able to:
1. Take a research question as input.
2. Use an LLM to generate a plan for answering the question, which may include calls to tools (e.g., web search, database query).
    2.1 The LLM should atleast get one result from Wikipedia using the search_topic tool.
3. Execute the plan by calling the appropriate tools and gathering information.
4. Summarize the findings and provide a final answer to the research question.
"""

import os
import json
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

# Researcher sub agent flow
"""
  Target flow

  START → orchestrator_node ──┬── researcher_a ──┐
                              ├── researcher_b ──┼── aggregator_node → summarizer_node → save_node → END
                              └── researcher_c ──┘

"""
# For persistent local storage uncomment the line below and install:
#   pip install langgraph-checkpoint-sqlite
# from langgraph.checkpoint.sqlite import SqliteSaver
# tools
from tools import wikipedia_search, web_search

load_dotenv()

class ResearchSummary(BaseModel):
    title: str          # one-line answer to the question
    bullets: list[str]  # 3 key facts
    confidence: str     # "high" | "medium" | "low"

class SubQueries(BaseModel):
    queries: list[str]  # exactly 3 focused sub-questions

class ResearchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int
    user_question: str
    summary: dict | None
    research_findings: list[str]
    research_sources: list[str]
    sub_queries: list[str]
    sub_results: Annotated[list[str], operator.add]  # appending reducer for parallel branches


llm = ChatLiteLLM(
    model="google/gemma-3-27b-it",
    api_base="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
    max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
    temperature=float(os.getenv("TEMPERATURE", 0.3)),
)

RESEARCHER_TOOLS = [wikipedia_search, web_search]


def build_research_agent_graph(llm: ChatLiteLLM) -> CompiledStateGraph:

    # ── Orchestrator node ──────────────────────────────────────────────────
    # Decomposes the user question into 3 focused sub-queries for parallel research.
    def orchestrator_node(state: ResearchAgentState):
        structured_llm = llm.with_structured_output(SubQueries)
        result = structured_llm.invoke([
            {"role": "system", "content": "Break this research question into exactly 3 focused sub-questions."},
            {"role": "user", "content": state["user_question"]},
        ])
        print(f"\n[orchestrator_node]: sub-queries={result.queries}")
        return {"sub_queries": result.queries}

    # ── Researcher nodes (fanout) ──────────────────────────────────────────
    # Factory that builds one researcher per sub-query index.
    # Each researcher: (1) asks the LLM which tool(s) to call, (2) executes those
    # tool calls, (3) returns findings to sub_results via the appending reducer.
    def make_researcher(index: int):
        llm_for_research = llm.bind_tools(RESEARCHER_TOOLS)

        def researcher_node(state: ResearchAgentState):
            query = state["sub_queries"][index]
            label = f"researcher_{chr(ord('a') + index)}"  # 0→'a', 1→'b', 2→'c' — used in print logs only

            # Step 1: LLM decides which tool(s) to call for this sub-query
            response = llm_for_research.invoke([
                {
                    "role": "system",
                    "content": (
                        "You are a focused research agent. Research the given sub-question "
                        "by calling wikipedia_search or web_search. You must call at least one tool."
                    ),
                },
                {"role": "user", "content": query},
            ])
            print(f"\n[{label}]: tool_calls={[tc['name'] for tc in response.tool_calls]}")

            # Step 2: Execute every tool call the LLM requested
            findings = []
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                if name == "wikipedia_search":
                    result = wikipedia_search.invoke(args)
                elif name == "web_search":
                    result = web_search.invoke(args)
                else:
                    result = f"Unknown tool: {name}"
                findings.append(result)

            # Fallback: if the LLM emitted no tool calls, search directly
            if not findings:
                findings.append(wikipedia_search.invoke({"query": query}))

            combined = "\n".join(findings)
            return {"sub_results": [f"[Sub-query {index + 1}: {query}]\n{combined}"]}

        return researcher_node

    # ── Aggregator node (fan-in) ───────────────────────────────────────────
    # LangGraph holds this node until all three researchers have written to
    # sub_results. Nothing to merge here — summarizer reads sub_results directly.
    def aggregator_node(state: ResearchAgentState):
        print(f"\n[aggregator_node]: {len(state['sub_results'])} sub-results ready")
        return {}

    # ── Summarizer node ────────────────────────────────────────────────────
    def summarizer_node(state: ResearchAgentState):
        combined = "\n\n".join(state["sub_results"]) if state["sub_results"] else "No research was conducted."

        system = {
            "role": "system",
            "content": (
                "You are given raw research results below. Summarize the key facts "
                "in 3 bullet points. Do not call any tools."
            ),
        }
        user = {"role": "user", "content": combined}

        result = llm.with_structured_output(ResearchSummary).invoke([system, user])
        summary = result.model_dump()
        print(f"\n[summarizer_node]: {summary}")
        return {"turn_count": state["turn_count"] + 1, "summary": summary}

    # ── Save node ──────────────────────────────────────────────────────────
    def save_node(state: ResearchAgentState):
        summary = state["summary"]
        with open("research_results.txt", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"\n[save_node]: results saved to research_results.txt")
        return {}

    # ── Graph wiring ───────────────────────────────────────────────────────
    graph = StateGraph(ResearchAgentState)

    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("researcher_a",      make_researcher(0))
    graph.add_node("researcher_b",      make_researcher(1))
    graph.add_node("researcher_c",      make_researcher(2))
    graph.add_node("aggregator_node",   aggregator_node)
    graph.add_node("summarizer_node",   summarizer_node)
    graph.add_node("save_node",         save_node)

    # fanout: orchestrator → 3 parallel researchers
    graph.add_edge(START,                "orchestrator_node")
    graph.add_edge("orchestrator_node",  "researcher_a")
    graph.add_edge("orchestrator_node",  "researcher_b")
    graph.add_edge("orchestrator_node",  "researcher_c")

    # fan-in: all 3 researchers → aggregator (LangGraph waits for all three)
    graph.add_edge("researcher_a",       "aggregator_node")
    graph.add_edge("researcher_b",       "aggregator_node")
    graph.add_edge("researcher_c",       "aggregator_node")

    graph.add_edge("aggregator_node",    "summarizer_node")
    graph.add_edge("summarizer_node",    "save_node")
    graph.add_edge("save_node",          END)

    return graph.compile(interrupt_before=["save_node"], checkpointer=MemorySaver())


if __name__ == "__main__":
    graph = build_research_agent_graph(llm)
    MAX_CYCLES = 7

    config = {"configurable": {"thread_id": "research-session-1"}}

    print("Research Agent ready. Type 'quit' or 'exit' to stop (max 7 questions).\n")

    for cycle in range(MAX_CYCLES):
        user_question = input(f"[{cycle + 1}/{MAX_CYCLES}] Enter your research question: ").strip()

        if user_question.lower() in ("quit", "exit"):
            print("Ending session.")
            break

        state = ResearchAgentState(
            messages=[HumanMessage(content=user_question)],
            turn_count=0,
            user_question=user_question,
            research_findings=[],
            research_sources=[],
            sub_queries=[],
            sub_results=[],
        )
        final_state = graph.invoke(state, config=config)

        print(final_state["summary"])
        save_summary = input("\nDo you want to save the research findings? (yes/no): ").strip().lower()
        if save_summary in ("yes", "y"):
            graph.invoke(None, config=config)
            print("Research findings saved.")
        elif save_summary in ("no", "n"):
            continue

        if cycle == MAX_CYCLES - 1:
            print("\nReached maximum of 7 questions. Ending session.")
