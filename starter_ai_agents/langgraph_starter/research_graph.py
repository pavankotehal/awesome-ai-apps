"""LangGraph ReAct agent — built manually without create_react_agent.

Flow:
  START → agent_node → (tool_calls?) → tools_node → agent_node → ... → END
"""
import os
import json
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from langchain_litellm import ChatLiteLLM
from langchain_core.tools import tool
from langchain_core.messages import messages_to_dict
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
# For persistent local storage uncomment the line below and install:
#   pip install langgraph-checkpoint-sqlite
# from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages

# ── State ──────────────────────────────────────────────────────────────────
# The State is a dict that gets passed between every node in the graph.
# Each node reads from it and returns a partial update.
# add_messages is a "reducer": instead of replacing messages, it appends.
class State(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int # example of another state variable with a reducer

# pydantic model to define the expected output of the summarizer_node. 
# This isn't strictly necessary, but it can help catch errors and clarify intent.
class SummaryResponse(BaseModel):
    summary: list[str] = Field(description="A list of 3 bullet points summarizing the research findings.")


load_dotenv()


@tool
def search_topic(query: str) -> str:
    """Search for information about a topic. Returns a short paragraph."""
    # fake it — just return f"Here is some info about {query}: ..."
    return f"Here is some info about {query}: ..."

@tool
def get_word_count(text: str) -> int:
    """Count words in a text."""
    return len(text.split())

@tool
def get_current_date() -> str:
    """Return today's date."""
    return datetime.now().strftime("%Y-%m-%d")

# All tools in one list — passed to both the LLM (so it knows what's available)
# and the ToolNode (so it can execute them).
TOOLS = [search_topic, get_word_count, get_current_date]

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
        # update the turn_count and messages in the state. messages uses the add_messages reducer, so it appends to the list instead of replacing it.
        return {"messages": [response],
                "turn_count": state["turn_count"] + 1,}  # add_messages reducer appends this

    def summarizer_node(state: State):
        system = {
            "role": "system",
            "content": (
                "You are a research assistant. Summarize the research findings "
                "from this conversation in 3 bullet points."
            ),
        }
        # with_structured_output forces the LLM to populate SummaryResponse.
        # Scoped only to this node — agent_node still returns free-form AIMessages.
        structured_llm = llm.with_structured_output(SummaryResponse)
        response: SummaryResponse = structured_llm.invoke([system] + state["messages"])
        # Join the bullet list into a single string for the message content
        content = "\n".join(f"• {point}" for point in response.summary)
        return {"messages": [AIMessage(content=content)]}
    
    def tools_condition_fn(state: State):
        if state["turn_count"] > 2:
          return "summarizer"  # force summarize after 3 turns
        # This condition function checks if the latest LLM response contains tool_calls.
        # If so, we route to the tools node. If not, we route to END.
        last_message = state["messages"][-1]

        # when Human asks "summarize", route to summarizer_node
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        # return bool(getattr(last_message, "tool_calls", []))
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        # if the message contains the word "summarize", route to summarizer_node
        elif human_messages and "summarize" in human_messages[-1].content.lower():
            return "summarizer"
        else:
            return END
    

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)       # LLM decision-making
    graph.add_node("summarizer", summarizer_node)  # LLM summarization
    graph.add_node("tools", ToolNode(TOOLS))  # tool execution (prebuilt, handles all tools)

    # ── Graph wiring ───────────────────────────────────────────────────────
    # Define edges between nodes. The agent_node can route to either tools_node or END.
    graph.add_edge(START, "agent")  # Always enter the graph at the agent
    graph.add_conditional_edges("agent", tools_condition_fn)  # Check if we need to call tools
    graph.add_edge("tools", "agent")  # After tools finish, always loop back to agent
    graph.add_edge("summarizer", END)  # After summarization, end the conversation


    # MemorySaver stores the full graph state (messages + turn_count) keyed by thread_id.
    # Swap for SqliteSaver("agent.db") to persist across restarts.
    checkpointer = MemorySaver()

    # compile() locks the graph and returns a runnable object
    return graph.compile(checkpointer=checkpointer)

# ── Main loop ──────────────────────────────────────────────────────────────

def main():
    agent = build_agent()
    print("🕸️  LangGraph agent ready. Type 'exit' to quit.\n")

    # thread_id identifies this conversation session.
    # The checkpointer stores all state (messages, turn_count) keyed by this ID.
    # Change it to start a fresh conversation; reuse it to resume a previous one.
    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not user:
            continue

        # Only pass the NEW user message — the checkpointer replays prior state automatically.
        # turn_count starts at 0 only for the very first message; after that the
        # checkpointer restores whatever value it saved at the end of the last turn.
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user}], "turn_count": 0},
            config=config,
        )

        # Persist full conversation state to JSON so you can inspect it.
        # messages_to_dict() converts LangChain message objects → plain dicts.
        memory = {
            "thread_id": thread_id,
            "turn_count": result["turn_count"],
            "messages": messages_to_dict(result["messages"]),
        }
        with open("memory.json", "w") as f:
            json.dump(memory, f, indent=2)

        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()