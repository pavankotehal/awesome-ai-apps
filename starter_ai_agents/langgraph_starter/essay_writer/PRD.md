# Essay Writer — Product Requirements Document

## Overview

A Streamlit UI on top of the LangGraph essay writer agent. The user enters a topic, optionally selects nodes to pause at, and watches the graph execute step by step — with the ability to inspect and edit intermediate state before the next node runs.

---

## Architecture

### Engine (`essay_writer.py`)
LangGraph graph with six nodes compiled with a `MemorySaver` checkpointer:

| Node | Input from state | Output to state |
|---|---|---|
| `plan` | `task` | `plan` (essay outline string) |
| `research` | `plan` | `queries` (list), `content` (list) |
| `generate` | `task`, `plan`, `content`, `critique` | `draft`, `revision_number` |
| `reflect` | `draft` | `critique` |
| `research_critique` | `critique` | `content` (appended) |
| `save` | `draft`, `task` | writes `essay_{slug}.md` to disk |

**Graph flow:**
```
START → plan → research → generate ──┬──(max revisions reached)──→ save → END
                            ↑         └──(revisions remaining)──→ reflect → research_critique ──┘
```

`build_digest_graph(interrupt_after: list[str] | None)` accepts a list of node names to pause at; uses `MemorySaver` as the checkpointer so state persists across stream calls.

### UI (`app.py`)
Streamlit app. Runs the graph via `graph.stream(..., stream_mode="updates")` and populates tabs as each node completes.

---

## Features

### 1. Topic input & configuration
- Free-text topic input field
- "Max Revisions" slider (1–5, default 2)

### 2. Node interrupt checkboxes
- One checkbox per node: Plan, Research, Generate, Reflect, Research Critique
- Checked nodes are passed to `build_digest_graph(interrupt_after=[...])` at run time
- Graph recompiles on each new run with the current checkbox selection

### 3. Per-node output tabs
Six tabs — Plan, Research, Generate, Reflect, Research Critique, Save — each populated as its node finishes:
- **Plan**: essay outline text
- **Research**: search queries + per-query findings
- **Generate**: each revision rendered under a `### Revision N` heading (accumulates across revisions)
- **Reflect**: critique text (accumulates across revisions)
- **Research Critique**: additional research findings added per revision
- **Save**: confirmation message with filename

Tab content is stored in `st.session_state.accumulated` and restored on every Streamlit rerun so output survives the page interaction cycle.

### 4. Interrupt & state editor
When a checked node completes, the graph pauses (`graph.get_state().next` is non-empty). A warning banner and an edit card appear below the tabs showing all populated state fields as editable text areas:

| Field | Editable as |
|---|---|
| `plan` | Multi-line text area |
| `queries` | One query per line |
| `draft` | Multi-line text area |
| `critique` | Multi-line text area |

**Continue →** calls `graph.update_state(thread, edited_values)` then resumes with `graph.stream(None, thread)`.  
**Cancel** returns to idle without resuming.

### 5. Run / Done lifecycle
- **Run** button disabled when topic is empty
- After completion a "Essay complete!" banner appears with a new Run button to start fresh (new thread ID, cleared accumulated state)
- Execution phase tracked in `st.session_state.phase`: `idle → running → interrupted → running → ... → done`

---

## Theme

Apple-inspired design enforced via two layers:

1. **`.streamlit/config.toml`** — sets Streamlit's primary color (`#0071e3`), background (`#f5f5f7`), secondary background (`#ffffff`), and text color (`#1d1d1f`) at the framework level (controls checkboxes, sliders, tab indicators)

2. **Injected CSS** — pill-shaped buttons, 12–18px border radius on inputs and cards, SF Pro / system font stack, `box-shadow: 0 2px 20px rgba(0,0,0,0.06)` for elevation, forced `background: white` and `-webkit-text-fill-color: #1d1d1f` on textareas to prevent dark-mode bleed

---

## File Structure

```
essay_writer/
├── app.py               # Streamlit UI
├── essay_writer.py      # LangGraph graph (build_digest_graph)
├── essay_prompts.py     # System prompts for each node
├── .streamlit/
│   └── config.toml      # Theme: Apple blue, light background
└── PRD.md               # This document
```

`tools.py` (wikipedia_search, web_search) lives in the parent `langgraph_starter/` directory and is resolved via `sys.path.insert(0, parent_dir)` in `app.py`.

---

## Running

```bash
# from langgraph_starter/
uv run streamlit run essay_writer/app.py
```

Debug logging (node-level): set `DEBUG=true` in environment before running.
