import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from essay_writer import build_digest_graph

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Essay Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Apple / Material CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
}

/* Force light background and dark text globally */
.stApp { background: #f5f5f7 !important; }
.stApp, .stApp * { color: #1d1d1f; }

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding: 3rem 4rem 2rem;
    max-width: 1140px;
}

/* Typography */
h1 {
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.03em !important;
    line-height: 1.08 !important;
    margin-bottom: 0 !important;
}
h2, h3 { color: #1d1d1f !important; font-weight: 600 !important; letter-spacing: -0.015em !important; }
p, li, span, div { color: #1d1d1f; }

/* Card-like containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important;
    border: 1px solid #e5e5ea !important;
    background: white !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.06) !important;
    padding: 0.25rem !important;
}

/* Buttons — Apple pill */
.stButton > button {
    background: #0071e3 !important;
    color: white !important;
    border: none !important;
    border-radius: 980px !important;
    padding: 0.55rem 1.6rem !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    transition: background 0.18s ease, transform 0.1s ease !important;
    box-shadow: none !important;
}
.stButton > button:hover:not(:disabled) {
    background: #0077ed !important;
    transform: scale(1.015) !important;
}
.stButton > button:disabled {
    background: #d2d2d7 !important;
    color: #8e8e93 !important;
}

/* Inputs */
.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 1.5px solid #d2d2d7 !important;
    background: white !important;
    padding: 0.7rem 1rem !important;
    font-size: 1rem !important;
    color: #1d1d1f !important;
    -webkit-text-fill-color: #1d1d1f !important;
    box-shadow: none !important;
    transition: border-color 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
    outline: none !important;
}
.stTextInput label, .stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #6e6e73 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Slider */
[data-testid="stSliderThumb"] {
    background: #0071e3 !important;
    border: 2.5px solid white !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.18) !important;
}
[data-testid="stSliderTrackActive"] {
    background: #0071e3 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: white !important;
    border-radius: 14px 14px 0 0 !important;
    padding: 0.4rem 0.8rem 0 !important;
    border-bottom: 1.5px solid #e5e5ea !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.55rem 1.1rem !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    color: #6e6e73 !important;
    border-radius: 8px 8px 0 0 !important;
    border-bottom: 2.5px solid transparent !important;
    margin-bottom: -1.5px !important;
    transition: color 0.15s !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #0071e3 !important;
    border-bottom-color: #0071e3 !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: white !important;
    border-radius: 0 0 14px 14px !important;
    padding: 1.75rem 2rem !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06) !important;
    min-height: 180px !important;
}

/* Checkboxes */
.stCheckbox > label {
    font-size: 0.88rem !important;
    color: #1d1d1f !important;
    cursor: pointer !important;
    gap: 6px !important;
}

/* Textarea — force white bg and dark text to prevent dark-mode bleed */
.stTextArea textarea,
textarea {
    border-radius: 12px !important;
    border: 1.5px solid #d2d2d7 !important;
    background: white !important;
    background-color: white !important;
    font-size: 0.88rem !important;
    color: #1d1d1f !important;
    -webkit-text-fill-color: #1d1d1f !important;
    line-height: 1.65 !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus,
textarea:focus {
    border-color: #0071e3 !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
    outline: none !important;
    background: white !important;
    background-color: white !important;
}
.stTextArea label {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #6e6e73 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Tab panel — force readable text color on all content */
.stTabs [data-baseweb="tab-panel"] p,
.stTabs [data-baseweb="tab-panel"] li,
.stTabs [data-baseweb="tab-panel"] span,
.stTabs [data-baseweb="tab-panel"] strong,
.stTabs [data-baseweb="tab-panel"] em,
.stTabs [data-baseweb="tab-panel"] div {
    color: #1d1d1f !important;
}

/* Markdown text everywhere */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.stMarkdown strong, .stMarkdown em, .stMarkdown span {
    color: #1d1d1f !important;
}

/* Alerts */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
    font-weight: 500 !important;
}

/* Status widget */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
    border: 1px solid #e5e5ea !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
}

/* Divider */
hr {
    border: none !important;
    border-top: 1px solid #e5e5ea !important;
    margin: 1.25rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
_NODES = ["plan", "research", "generate", "reflect", "research_critique"]
_EMPTY_ACCUM = {n: "" for n in [*_NODES, "save"]}

# ── Session state ──────────────────────────────────────────────────────────────
_defaults = {
    "phase": "idle",          # idle | running | interrupted | done
    "graph": None,
    "thread": None,
    "initial_input": None,    # task dict on first run, None on resume
    "accumulated": dict(_EMPTY_ACCUM),
    "prev_content_len": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# Essay Writer")
st.markdown(
    "<p style='font-size:1.1rem;color:#6e6e73;margin-top:-0.4rem;margin-bottom:2rem;"
    "letter-spacing:-0.01em'>Research. Draft. Refine. Automatically.</p>",
    unsafe_allow_html=True,
)

# ── Controls ───────────────────────────────────────────────────────────────────
with st.container(border=True):
    col_topic, col_rev = st.columns([3, 1])
    with col_topic:
        topic = st.text_input("Topic", placeholder="e.g. The impact of AI on modern healthcare")
    with col_rev:
        max_revisions = st.slider("Max Revisions", 1, 5, 2)

    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;color:#6e6e73;"
        "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem'>"
        "Pause &amp; edit after node</p>",
        unsafe_allow_html=True,
    )
    cb_cols = st.columns(len(_NODES))
    interrupt_nodes = []
    for col, node in zip(cb_cols, _NODES):
        with col:
            label = node.replace("_", " ").title()
            if st.checkbox(label, key=f"cb_{node}"):
                interrupt_nodes.append(node)

# ── Tabs ───────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

tab_plan, tab_research, tab_generate, tab_reflect, tab_rc, tab_save = st.tabs([
    "Plan", "Research", "Generate", "Reflect", "Research Critique", "Save",
])

placeholders = {
    "plan": tab_plan.empty(),
    "research": tab_research.empty(),
    "generate": tab_generate.empty(),
    "reflect": tab_reflect.empty(),
    "research_critique": tab_rc.empty(),
    "save": tab_save.empty(),
}

# Restore tab content on every rerun
for node, content in st.session_state.accumulated.items():
    if content and node in placeholders:
        placeholders[node].markdown(content)

# ── Interrupt editor ───────────────────────────────────────────────────────────
if st.session_state.phase == "interrupted":
    current = st.session_state.graph.get_state(st.session_state.thread)
    next_node = current.next[0] if current.next else "unknown"
    values = current.values

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.warning(
        f"Paused — reviewing state before **{next_node}** runs. "
        "Edit any field below, then continue.",
        icon="✏️",
    )

    with st.container(border=True):
        st.markdown("### Edit State")

        edited = {}

        if values.get("plan"):
            edited["plan"] = st.text_area("Plan", value=values["plan"], height=220)

        if values.get("queries"):
            raw = st.text_area(
                "Queries (one per line)",
                value="\n".join(values["queries"]),
                height=120,
            )
            edited["queries"] = [q.strip() for q in raw.splitlines() if q.strip()]

        if values.get("draft"):
            edited["draft"] = st.text_area("Draft", value=values["draft"], height=320)

        if values.get("critique"):
            edited["critique"] = st.text_area("Critique", value=values["critique"], height=220)

        col_cont, col_cancel, _ = st.columns([1, 1, 6])
        with col_cont:
            if st.button("Continue →"):
                if edited:
                    st.session_state.graph.update_state(st.session_state.thread, edited)
                st.session_state.initial_input = None
                st.session_state.phase = "running"
                st.rerun()
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.phase = "idle"
                st.rerun()

# ── Run / Done ─────────────────────────────────────────────────────────────────
if st.session_state.phase in ("idle", "done"):
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    if st.session_state.phase == "done":
        st.success("Essay complete! Start a new run below.", icon="✅")

    if st.button("Run", disabled=not topic.strip()):
        st.session_state.graph = build_digest_graph(interrupt_after=interrupt_nodes)
        st.session_state.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        st.session_state.accumulated = dict(_EMPTY_ACCUM)
        st.session_state.prev_content_len = 0
        st.session_state.initial_input = {
            "task": f"Write an essay about {topic}",
            "max_revisions": max_revisions,
            "revision_number": 0,
        }
        st.session_state.phase = "running"
        st.rerun()

# ── Streaming loop ─────────────────────────────────────────────────────────────
if st.session_state.phase == "running":
    graph = st.session_state.graph
    thread = st.session_state.thread
    input_data = st.session_state.initial_input
    st.session_state.initial_input = None  # clear so next rerun resumes with None

    with st.status("Running essay writer…", expanded=True) as status:
        for update in graph.stream(input_data, thread, stream_mode="updates"):
            for node_name, node_output in update.items():
                if node_name not in placeholders:
                    continue

                st.write(f"Completed → **{node_name}**")
                acc = st.session_state.accumulated

                if node_name == "plan":
                    acc["plan"] = node_output.get("plan", "")

                elif node_name == "research":
                    queries = node_output.get("queries", [])
                    pieces = node_output.get("content", [])
                    lines = ["**Search Queries**", ""]
                    lines += [f"- {q}" for q in queries]
                    lines += ["", "---", "**Research Findings**", ""]
                    for i, piece in enumerate(pieces, 1):
                        lines += [f"**Query {i}:**", piece, ""]
                    acc["research"] = "\n".join(lines)
                    st.session_state.prev_content_len = len(pieces)

                elif node_name == "generate":
                    rev = node_output.get("revision_number", "")
                    acc["generate"] += f"\n\n---\n\n### Revision {rev}\n\n{node_output.get('draft', '')}"

                elif node_name == "reflect":
                    acc["reflect"] += f"\n\n---\n\n{node_output.get('critique', '')}"

                elif node_name == "research_critique":
                    all_content = node_output.get("content", [])
                    new_pieces = all_content[st.session_state.prev_content_len:]
                    st.session_state.prev_content_len = len(all_content)
                    lines = [f"**Additional research — {len(new_pieces)} piece(s)**", ""]
                    for piece in new_pieces:
                        lines += [piece, ""]
                    acc["research_critique"] += "\n".join(lines)

                elif node_name == "save":
                    acc["save"] = "✅  Essay saved to file."

                placeholders[node_name].markdown(acc[node_name])

        # Detect interrupt vs. completion
        current_state = graph.get_state(thread)
        if current_state.next:
            status.update(
                label=f"Paused — ready to edit before **{current_state.next[0]}**",
                state="error",
            )
            st.session_state.phase = "interrupted"
        else:
            status.update(label="Done!", state="complete")
            st.session_state.phase = "done"

    st.rerun()
