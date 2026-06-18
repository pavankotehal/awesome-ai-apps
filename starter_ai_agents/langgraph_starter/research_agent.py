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
from datetime import datetime
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_litellm import ChatLiteLLM
from langgraph import graph
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
# For persistent local storage uncomment the line below and install:
#   pip install langgraph-checkpoint-sqlite
# from langgraph.checkpoint.sqlite import SqliteSaver
# tools
from tools import wikipedia_search, web_search, save_research_findings

load_dotenv()

class ResearchAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int
    user_question: str
    research_findings: list[str]
    research_sources: list[str]

llm = ChatLiteLLM(
        model="openai/deepseek-ai/DeepSeek-V3.2",
        api_base="https://api.tokenfactory.nebius.com/v1/",
        api_key=os.environ["NEBIUS_API_KEY"],
        max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
        temperature=float(os.getenv("TEMPERATURE", 0.3)),
    )


# define tools here, e.g. search_topic, get_word_count, get_current_date, etc.
# we will use tools such as wikipedia, duckduckgo, websearch_tool.
TOOLS = [wikipedia_search, web_search, save_research_findings]

def build_research_agent_graph(llm: ChatLiteLLM) -> CompiledStateGraph:
    
    # bind_tools tells the LLM about the available tools (names, descriptions, params).
    # The LLM won't execute them — it just signals WHICH tool to call via tool_calls.
    llm_with_tools = llm.bind_tools(TOOLS)

    # ── Agent node ─────────────────────────────────────────────────────────
    # This node runs every time the LLM needs to make a decision.
    # It receives the full message history, calls the LLM, and returns the response.
    #
    # The LLM response is one of two things:
    #   AIMessage(tool_calls=[...])  → LLM wants to call a tool → route to tools node
    #   AIMessage(content="...")     → LLM has a final answer  → route to END
    #
    # agent_node is defined inside build_agent so it can close over llm_with_tools.
    def agent_node(state: ResearchAgentState):
        system = {
            "role": "system",
            "content": (
                 "You are a research assistant. You MUST use the wikipedia_search tool "
                "to research every question before answering. Never answer from memory alone. "
                "save_research_results with the question and your final summary before finishing."
            ),
        }
        response = llm_with_tools.invoke([system] + state["messages"])
        if response.content:
            print(f"\n[agent_node]: {response.content}")
        if response.tool_calls:
            names = [tc["name"] for tc in response.tool_calls]
            print(f"\n[agent_node]: calling {names}")
        return {"messages": [response],
                "turn_count": state["turn_count"] + 1,}  # add_messages reducer appends this
    
    # ── Summarizer node ─────────────────────────────────────────────────────────
    def summarizer_node(state: ResearchAgentState):
        tool_results = [
            msg.content for msg in state["messages"]
            if isinstance(msg, ToolMessage)
        ]
        combined = "\n\n".join(tool_results) if tool_results else "No research was conducted."

        system = {
            "role": "system",
            "content": (
                "You are given raw research results below. Summarize the key facts "
                "in 3 bullet points. Do not call any tools."
            ),
        }
        user = {"role": "user", "content": combined}

        response = llm.invoke([system, user])
        print(f"\n[summarizer_node]: {response.content}")
        return {"messages": [response],
                "turn_count": state["turn_count"] + 1,}  # add_messages reducer appends this
    
    # save node
    def save_node(state: ResearchAgentState):
        summary = state["messages"][-1].content
        result = {
            "timestamp": datetime.now().isoformat(),
            "question": state["user_question"],
            "summary": summary
        }
        with open("research_results.txt", "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"\n[save_node]: results saved to research_results.txt")
        return {}
    

    # conditional routing function to determine whether to call a tool or end the conversation
    def route_node(state: ResearchAgentState):
        # if message has tool_calls, route to tool_node, 
        # otherwise if count exceeds limit, route to summarizer_node
        # if no tools and summarizer already called, route to END
        # count less than 2, route to agent_node to continue the conversation
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls and state["turn_count"] < 5:
            return "tool_node"  # route to tool execution node
        else:
            return "summarizer_node"  # route to summarizer node for final answer
    
    
    graph = StateGraph(ResearchAgentState)
    
    graph.add_node("agent_node", agent_node)
    graph.add_node("summarizer_node", summarizer_node)
    graph.add_node("tool_node", ToolNode(TOOLS))
    graph.add_node("save_node", save_node)
    # define edges
    graph.add_edge(START, "agent_node")

    graph.add_conditional_edges(
      "agent_node",
      route_node,
      {"tool_node": "tool_node", "summarizer_node": "summarizer_node"}
    )
    graph.add_edge("tool_node", "agent_node")
    graph.add_edge("summarizer_node", "save_node")
    graph.add_edge("save_node", END)

    
    # Define the nodes and edges of the graph here, using the tools and LLM as needed.
    # For example:
    # - Start node takes user input and initializes state.
    # - Agent node generates a plan using the LLM, which may include tool calls.
    # - Tool nodes execute the tools and update the state with findings.
    # - Summarizer node takes all findings and generates a final summary.

    return graph.compile(checkpointer=MemorySaver())  # Use MemorySaver for in-memory checkpointing

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
            messages=HumanMessage(content=user_question),
            turn_count=0,
            user_question=user_question,
            research_findings=[],
            research_sources=[],
        )
        final_state = graph.invoke(state, config=config)
        # messages = final_state["messages"]

        if cycle == MAX_CYCLES - 1:
            print("\nReached maximum of 7 questions. Ending session.")