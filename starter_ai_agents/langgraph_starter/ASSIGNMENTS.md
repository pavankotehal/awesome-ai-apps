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

**Status:** ✅ Done

---

### Task 2 — Persistent memory with MemorySaver
**Concept:** Checkpointers + `thread_id`

**Status:** ✅ Done

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

**Status:** ✅ Done

---

### Session 2 completion checklist

| Task | Signal | Status |
|---|---|---|
| Streaming | Terminal output appears incrementally, not all at once | ✅ |
| MemorySaver | Manual `messages` carry-over removed, context still works | ✅ |
| Human-in-loop | Agent pauses before saving, respects y/n | ✅ |

---

## Session 3 — 2026-06-19

**Goal:** Finish the remaining Session 2 task, then learn three patterns that real production agents use — structured output, parallel execution, and sub-graphs. End with a small real-world project that ties everything together.

**File:** `research_agent.py` for Tasks 3–5. New file `digest_agent.py` for Task 6.

---

### Task 3 (carry-over) — Complete human-in-the-loop

**Status:** ✅ Done

---

### Task 4 — Structured output with Pydantic
**Concept:** `llm.with_structured_output(Model)`

Right now `summarizer_node` returns a free-form string. Real agents need typed, predictable output — so downstream code can reliably read fields without parsing text.

**What to do:**
1. Define a Pydantic model for the summary:
   ```python
   from pydantic import BaseModel

   class ResearchSummary(BaseModel):
       title: str           # one-line answer to the question
       bullets: list[str]   # 3 key facts
       confidence: str      # "high" | "medium" | "low"
   ```
2. In `summarizer_node`, replace `llm.invoke(...)` with `llm.with_structured_output(ResearchSummary).invoke(...)`
3. The response is now a `ResearchSummary` object, not a string. Update `save_node` to write `result.model_dump()` instead of `summary`
4. Update the print to show each field cleanly

**Verify:** The summarizer prints a structured object. `research_results.txt` contains JSON with `title`, `bullets`, `confidence` fields.

**Key thing to understand:** `with_structured_output` makes the LLM return a validated Pydantic object every time. This is how you make LLM output reliable enough to use in real code.

**Status:** ✅ Done

---

### Task 5 — Parallel tool calls with the Send API
**Concept:** `Send`, fan-out / fan-in pattern

Right now the agent researches one question at a time, sequentially. The `Send` API lets you fan out work across multiple parallel nodes — a core pattern in multi-agent systems.

**What to do:**
1. Add a `topics: list[str]` field to `ResearchAgentState`
2. Add a `fan_out_node` that returns a list of `Send` objects — one per topic:
   ```python
   from langgraph.types import Send

   def fan_out_node(state):
       return [Send("agent_node", {**state, "user_question": t, "messages": [HumanMessage(content=t)]})
               for t in state["topics"]]
   ```
3. Wire `START → fan_out_node` with a conditional edge that routes to `fan_out_node` if `topics` is set, else to `agent_node` as before
4. Each parallel branch runs its own `agent_node → tool_node → summarizer_node` pipeline

**Verify:** Pass `topics=["topic A", "topic B"]` in state. You see two parallel research chains running, each printing their own `[agent_node]` / `[summarizer_node]` lines.

**Key thing to understand:** `Send(node_name, state)` schedules a node invocation with a custom state. LangGraph runs all `Send` targets in parallel. This is how you build fan-out patterns without polling or threads.

**Status:** [ ] Not started

---

### Task 6 — Real-world mini project: Daily Digest Agent
**Concept:** Applying everything in one coherent agent

Build a new agent in `digest_agent.py` that produces a daily research digest on a list of topics you care about.

**What it should do:**
1. Accept a hardcoded list of topics at startup (e.g. `["AI news", "Python releases", "space exploration"]`)
2. Research each topic in parallel using the `Send` fan-out pattern (Task 5)
3. Each branch produces a `ResearchSummary` (Task 4)
4. A `compile_digest_node` collects all summaries and formats them into a markdown report
5. Pause before saving and ask for confirmation (Task 3's interrupt pattern)
6. Save the report to `digest_YYYY-MM-DD.md`

**State shape to aim for:**
```python
class DigestState(TypedDict):
    topics: list[str]
    summaries: Annotated[list, operator.add]   # fan-in: each branch appends its summary
    report: str
```

**Verify:** Running `python digest_agent.py` researches all topics in parallel, prints the digest, asks `Save? (y/n)`, and writes a dated markdown file on `y`.

**Key thing to understand:** This is a complete, real agent — parallel research, typed output, human approval, persistent output. The same pattern applies to any multi-source research task (competitor monitoring, news digest, report generation).

**Status:** [ ] Not started

---

### Session 3 completion checklist

| Task | Concept | Signal | Status |
|---|---|---|---|
| Task 3 | Human-in-loop | Graph pauses, y saves, n skips | ✅ |
| Task 4 | Structured output | Summarizer returns a Pydantic object | ✅ |
| Task 5 | Send / fan-out | Two topics research in parallel | [ ] |
| Task 6 | Real-world project | `digest_agent.py` runs end-to-end | [ ] |
