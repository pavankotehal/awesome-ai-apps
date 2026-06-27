# Langgraph agent to write the essay
# nodes:
#   plan : generate essay outline
#   research: generate queries from plan + call tools to gather content
#   generate: write essay draft
#   reflect: critique the draft
#   research_critique: gather more content based on critique

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from essay_prompts import PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_PLAN_PROMPT, RESEARCH_CRITIQUE_PROMPT
from tools import wikipedia_search, web_search

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

def _log(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)


class Sub_queries(BaseModel):
    queries: list[str]


class AgentState(TypedDict):
    task: str
    plan: str
    queries: list[str]
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


llm = ChatLiteLLM(
    model="openai/Qwen/Qwen3-235B-A22B-Instruct-2507",
    api_base=os.environ["NEBIUS_BASE_URL"],
    api_key=os.environ["NEBIUS_API_KEY"],
    max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
    temperature=float(os.getenv("TEMPERATURE", 0.3)),
)

RESEARCHER_TOOLS = [wikipedia_search, web_search]


def build_digest_graph() -> CompiledStateGraph:

    def plan_node(state: AgentState):
        _log(f"plan: generating outline for task='{state['task']}'")
        response = llm.invoke([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": state["task"]},
        ])
        _log(f"plan: outline generated ({len(response.content)} chars)")
        _log(f"plan: {response.content}")
        return {"plan": response.content}

    def research_node(state: AgentState):
        _log("research: generating search queries from plan")
        llm_structured = llm.with_structured_output(Sub_queries)
        queries_response = llm_structured.invoke([
            {"role": "system", "content": RESEARCH_PLAN_PROMPT},
            {"role": "user", "content": state["plan"]},
        ])
        queries = queries_response.queries
        _log(f"research: generated {len(queries)} queries: {queries}")

        llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
        tool_map = {t.name: t for t in RESEARCHER_TOOLS}
        content = []

        for query in queries:
            _log(f"research: searching for '{query}'")
            messages = [
                {"role": "system", "content": "Use the available tools to research the following query."},
                {"role": "user", "content": query},
            ]
            response = llm_with_tools.invoke(messages)
            while response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    _log(f"research: calling tool '{tc['name']}' with args={tc['args']}")
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                response = llm_with_tools.invoke(messages)
            content.append(response.content)

        _log(f"research: collected {len(content)} content pieces")
        return {"queries": queries, "content": content}

    def generate_node(state: AgentState):
        revision = state.get("revision_number", 0) + 1
        _log(f"generate: writing draft (revision {revision}/{state['max_revisions']})")
        content_str = "\n\n".join(state.get("content", []))
        user_message = f"{state['task']}\n\n---\n\nPlan:\n{state['plan']}"
        if state.get("critique"):
            user_message += f"\n\nCritique:\n{state['critique']}"
        response = llm.invoke([
            {"role": "system", "content": WRITER_PROMPT.format(content=content_str)},
            {"role": "user", "content": user_message},
        ])
        _log(f"generate: draft complete ({len(response.content)} chars)")
        return {
            "draft": response.content,
            "revision_number": revision,
        }

    def reflect_node(state: AgentState):
        _log("reflect: critiquing draft")
        response = llm.invoke([
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": state["draft"]},
        ])
        _log(f"reflect: critique complete ({len(response.content)} chars)")
        return {"critique": response.content}

    def research_critique_node(state: AgentState):
        _log("research_critique: generating queries from critique")
        llm_structured = llm.with_structured_output(Sub_queries)
        queries_response = llm_structured.invoke([
            {"role": "system", "content": RESEARCH_CRITIQUE_PROMPT},
            {"role": "user", "content": state["critique"]},
        ])
        _log(f"research_critique: queries={queries_response.queries}")

        llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
        tool_map = {t.name: t for t in RESEARCHER_TOOLS}
        content = list(state.get("content", []))

        for query in queries_response.queries:
            _log(f"research_critique: searching for '{query}'")
            messages = [
                {"role": "system", "content": "Use the available tools to research the following query."},
                {"role": "user", "content": query},
            ]
            response = llm_with_tools.invoke(messages)
            while response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    _log(f"research_critique: calling tool '{tc['name']}' with args={tc['args']}")
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                response = llm_with_tools.invoke(messages)
            content.append(response.content)

        _log(f"research_critique: total content pieces now {len(content)}")
        return {"content": content}

    def should_continue(state: AgentState):
        if state["revision_number"] >= state["max_revisions"]:
            return "save"
        return "reflect"

    def save_node(state: AgentState):
        slug = state["task"].lower().replace(" ", "_")[:50]
        filename = f"essay_{slug}.md"
        with open(filename, "w") as f:
            f.write(f"# {state['task']}\n\n")
            f.write(state["draft"])
        _log(f"save: essay written to '{filename}'")
        print(f"\nEssay saved to {filename}")
        return {}

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("research_critique", research_critique_node)
    builder.add_node("save", save_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "generate")
    builder.add_edge("research_critique", "generate")
    builder.add_conditional_edges("generate", should_continue, {"save": "save", "reflect": "reflect"})
    builder.add_edge("reflect", "research_critique")
    builder.add_edge("save", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    graph = build_digest_graph()
    thread = {"configurable": {"thread_id": "essay-1"}}
    user_input = input("Please type the topic you want to research and write essay about..\n")
    result = graph.invoke(
        {
            "task": f"Write an essay about {user_input}",
            "max_revisions": 2,
            "revision_number": 0,
        },
        thread,
    )
    print(result["draft"])
