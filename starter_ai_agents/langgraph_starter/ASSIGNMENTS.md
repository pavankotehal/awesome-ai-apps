# LangGraph Learning Assignments

---

## Session 1 — 2026-06-10 ✅

**Concepts covered:**
- LangGraph core primitives: State, Nodes, Edges
- `add_messages` reducer
- `agent_node` closing over `llm_with_tools`
- Manual routing with `tools_condition_fn`
- `ToolNode` prebuilt executor
- Graph wiring: `add_edge`, `add_conditional_edges`
- Multi-turn memory in `main()` — accumulating messages across turns

**Files built:**
- `new_graph.py` — manual ReAct agent (2 nodes: agent + tools)
- `research_graph.py` — 3-node graph (agent + tools + summarizer)

**Assignment:** Build a 3-node research graph with manual routing, `turn_count` guard, and a summarizer node that uses plain `llm` (not `llm_with_tools`). ✅

---

## Session 2 — 2026-06-17

**File:** `research_agent.py`

Three tasks, each teaching a new LangGraph primitive. Do them in order.

---

### Task 1 — Streaming output
**Concept:** `graph.stream()` instead of `graph.invoke()`

Right now the agent is silent until it finishes. Fix that.

**What to do:**
- In the `__main__` loop, replace `graph.invoke(state)` with `graph.stream(state)`
- Print each chunk as it arrives. LangGraph streams events as dicts — key is the node name, value is the partial state update.
- You should see output appear token-by-token (or node-by-node) instead of waiting for the full answer.

**Verify:** You see `agent_node`, `tool_node`, `summarizer_node` output appearing progressively in the terminal, not all at once.

**Key thing to understand:** `stream()` yields `{node_name: state_update}` dicts. You'll need to decide what to print from each.

**Status:** [ ] Not started

---

### Task 2 — Persistent memory with MemorySaver
**Concept:** Checkpointers + `thread_id`

Right now you manually carry `messages` between cycles. LangGraph can do this for you.

**What to do:**
1. Import `MemorySaver` from `langgraph.checkpoint.memory`
2. Pass it to `graph.compile(checkpointer=MemorySaver())`
3. In `__main__`, create a single `config = {"configurable": {"thread_id": "research-session-1"}}`
4. Pass that config to every `graph.invoke(state, config=config)` call
5. Remove the manual `messages = final_state["messages"]` carry-over — the checkpointer handles it

**Verify:** On the second question, the agent still has the first question in its history — but you didn't wire it manually.

**Key thing to understand:** A `thread_id` is like a conversation ID. The checkpointer snapshots state after every node. On the next invoke, it restores from the last checkpoint for that thread.

**Status:** [ ] Not started

---

### Task 3 — Human-in-the-loop before saving
**Concept:** `interrupt_before`, `Command(resume=...)`

Right now the agent always saves results without asking. Add a confirmation step.

**What to do:**
1. Add `interrupt_before=["save_node"]` to `graph.compile(...)` (requires Task 2's checkpointer)
2. In `__main__`, after `graph.invoke()` returns mid-graph (it'll stop before `save_node`), print the summary and ask: `Save this result? (y/n): `
3. If `y` → call `graph.invoke(None, config=config)` to resume (passing `None` resumes from checkpoint)
4. If `n` → skip the second invoke entirely

**Verify:** After the agent summarizes, it pauses and waits for your input. Typing `n` skips the file write; typing `y` resumes and saves.

**Key thing to understand:** `interrupt_before` makes the graph pause *before* that node runs. The state is checkpointed at that point. Calling `invoke(None, config)` resumes from that exact checkpoint.

**Status:** [ ] Not started

---

### Session 2 completion checklist

| Task | Signal | Status |
|---|---|---|
| Streaming | Terminal output appears incrementally, not all at once | [ ] |
| MemorySaver | Manual `messages` carry-over removed, context still works | [ ] |
| Human-in-loop | Agent pauses before saving, respects y/n | [ ] |
