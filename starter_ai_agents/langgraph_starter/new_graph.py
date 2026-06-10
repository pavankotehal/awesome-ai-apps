"""LangGraph ReAct agent — built manually without create_react_agent.

Flow:
  START → agent_node → (tool_calls?) → tools_node → agent_node → ... → END
"""
import os
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

# ── State ──────────────────────────────────────────────────────────────────
# The State is a dict that gets passed between every node in the graph.
# Each node reads from it and returns a partial update.
# add_messages is a "reducer": instead of replacing messages, it appends.
class State(TypedDict):
    messages: Annotated[list, add_messages]


load_dotenv()


# ── Tools ──────────────────────────────────────────────────────────────────
# @tool turns a plain function into a LangChain tool.
# The docstring becomes the tool description the LLM reads to decide when to use it.

@tool
def get_current_time() -> str:
    """Return the current local date and time as an ISO-8601 string."""
    return datetime.now().isoformat(timespec="seconds")


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in the given text."""
    return len(text.split())


@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def get_probnorm_value(probability: float) -> float:
    """Convert a probability value to a probnorm value."""
    from scipy.stats import norm
    return norm.cdf(probability)


# All tools in one list — passed to both the LLM (so it knows what's available)
# and the ToolNode (so it can execute them).
TOOLS = [get_probnorm_value, add_numbers, get_current_time, word_count]


# ── Build graph ────────────────────────────────────────────────────────────

def build_agent():
    llm = ChatLiteLLM(
        model="openai/deepseek-ai/DeepSeek-V3.2",
        api_base="https://api.tokenfactory.nebius.com/v1/",
        api_key=os.environ["NEBIUS_API_KEY"],
        max_tokens=int(os.getenv("MAX_TOKENS", 1024)),
        temperature=float(os.getenv("TEMPERATURE", 0.3)),
    )

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
    def agent_node(state: State):
        system = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools when they are relevant "
                "instead of guessing."
            ),
        }
        # Send the full conversation history on every call — LLM has no memory between calls.
        response = llm_with_tools.invoke([system] + state["messages"])
        return {"messages": [response]}  # add_messages reducer appends this

    # ── Graph wiring ───────────────────────────────────────────────────────
    graph = StateGraph(State)

    # Register the two nodes
    graph.add_node("agent", agent_node)       # LLM decision-making
    graph.add_node("tools", ToolNode(TOOLS))  # tool execution (prebuilt, handles all tools)

    # Always enter the graph at the agent node
    graph.add_edge(START, "agent")

    # After agent_node: tools_condition checks if the last AIMessage has tool_calls.
    #   tool_calls present → route to "tools"
    #   tool_calls empty   → route to END
    graph.add_conditional_edges("agent", tools_condition)

    # After tools finish, always loop back to agent so it can process the results
    graph.add_edge("tools", "agent")

    # compile() locks the graph and returns a runnable object
    return graph.compile()


# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    agent = build_agent()
    print("🕸️  LangGraph ReAct agent ready. Type 'exit' to quit.\n")

    # messages accumulates the full conversation across turns (multi-turn memory)
    messages = []

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not user:
            continue

        # Append the new user message to history
        messages.append({"role": "user", "content": user})

        # invoke() runs the full graph (may loop agent→tools→agent multiple times)
        # and returns the final state once the graph reaches END
        result = agent.invoke({"messages": messages})

        # result["messages"] contains ALL messages including tool calls/results.
        # Replace messages so next turn has full context.
        messages = result["messages"]

        # The last message is always the final AIMessage with the answer
        print(f"\nAgent: {messages[-1].content}\n")


if __name__ == "__main__":
    main()
