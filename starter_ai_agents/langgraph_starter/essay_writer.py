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
        response = llm.invoke([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": state["task"]},
        ])
        return {"plan": response.content}

    def research_node(state: AgentState):
        # Generate search queries from the essay plan
        llm_structured = llm.with_structured_output(Sub_queries)
        queries_response = llm_structured.invoke([
            {"role": "system", "content": RESEARCH_PLAN_PROMPT},
            {"role": "user", "content": state["plan"]},
        ])
        queries = queries_response.queries

        # Call tools for each query and collect content
        llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
        tool_map = {t.name: t for t in RESEARCHER_TOOLS}
        content = []

        for query in queries:
            messages = [
                {"role": "system", "content": "Use the available tools to research the following query."},
                {"role": "user", "content": query},
            ]
            response = llm_with_tools.invoke(messages)
            while response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                response = llm_with_tools.invoke(messages)
            content.append(response.content)

        return {"queries": queries, "content": content}

    def generate_node(state: AgentState):
        content_str = "\n\n".join(state.get("content", []))
        user_message = f"{state['task']}\n\n---\n\nPlan:\n{state['plan']}"
        if state.get("critique"):
            user_message += f"\n\nCritique:\n{state['critique']}"
        response = llm.invoke([
            {"role": "system", "content": WRITER_PROMPT.format(content=content_str)},
            {"role": "user", "content": user_message},
        ])
        return {
            "draft": response.content,
            "revision_number": state.get("revision_number", 0) + 1,
        }

    def reflect_node(state: AgentState):
        response = llm.invoke([
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": state["draft"]},
        ])
        return {"critique": response.content}

    def research_critique_node(state: AgentState):
        llm_structured = llm.with_structured_output(Sub_queries)
        queries_response = llm_structured.invoke([
            {"role": "system", "content": RESEARCH_CRITIQUE_PROMPT},
            {"role": "user", "content": state["critique"]},
        ])

        llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
        tool_map = {t.name: t for t in RESEARCHER_TOOLS}
        content = list(state.get("content", []))

        for query in queries_response.queries:
            messages = [
                {"role": "system", "content": "Use the available tools to research the following query."},
                {"role": "user", "content": query},
            ]
            response = llm_with_tools.invoke(messages)
            while response.tool_calls:
                messages.append(response)
                for tc in response.tool_calls:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                response = llm_with_tools.invoke(messages)
            content.append(response.content)

        return {"content": content}

    def should_continue(state: AgentState):
        if state["revision_number"] >= state["max_revisions"]:
            return END
        return "reflect"

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("research", research_node)
    builder.add_node("generate", generate_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("research_critique", research_critique_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("research", "generate")
    builder.add_edge("research_critique", "generate")
    builder.add_conditional_edges("generate", should_continue, {END: END, "reflect": "reflect"})
    builder.add_edge("reflect", "research_critique")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    graph = build_digest_graph()
    thread = {"configurable": {"thread_id": "essay-1"}}
    user_input = input("Please type the topic you want to research and write essay about..\n")

    current_node = None
    for chunk, metadata in graph.stream(
        {
            "task": f"Write an essay about the {user_input}",
            "max_revisions": 2,
            "revision_number": 0,
        },
        thread,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")
        if node != current_node:
            current_node = node
            print(f"\n\n{'='*50}\n[{node}]\n{'='*50}\n")
        if hasattr(chunk, "content") and chunk.content:
            print(chunk.content, end="", flush=True)

    print()
