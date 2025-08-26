# # jd_agent.py
# import os
# from typing import Optional
# from dotenv import load_dotenv
# import streamlit as st

# # LangChain + OpenAI
# try:
#     # Modern import path
#     from langchain_openai import ChatOpenAI
# except Exception:
#     # Fallback if older LC is installed
#     from langchain.chat_models import ChatOpenAI  # type: ignore

# load_dotenv()  # Loads OPENAI_API_KEY and anything else

# DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# SYSTEM_PROMPT = """You are a concise technical recruiter and hiring manager helper.
# Given a few structured inputs (role, must-have skills, nice-to-have, experience, location, notes),
# draft a clear, scannable job description with:
# - Role summary (3–5 lines, no fluff)
# - Responsibilities (5–8 bullets, action verbs)
# - Must-have skills (bullet list)
# - Good-to-have skills (bullet list)
# - Location & work type (onsite/hybrid/remote)
# - Experience range
# - Application/next steps

# Keep it professional and to the point. Avoid over-selling with superlatives.
# """

# DRAFT_TEMPLATE = """Role: {role}
# Location: {location}
# Experience: {experience}
# Must-have skills (comma-separated): {must_have}
# Good-to-have skills (comma-separated): {nice_to_have}
# Additional notes: {notes}

# Now produce the full job description as markdown, with clear section headers and bullet lists where relevant.
# """

# REFINE_TEMPLATE = """Here is the current JD draft:

# ---
# {current_jd}
# ---

# User feedback/refinement instructions:
# {feedback}

# Revise the JD accordingly, keeping the same structure and concision. Output the full revised JD as markdown only.
# """

# def _get_llm(temperature: float = 0.4) -> ChatOpenAI:
#     # OPENAI_API_KEY must be present in the environment; load_dotenv() already called.
#     return ChatOpenAI(model=DEFAULT_MODEL, temperature=temperature)

# def _draft_jd(role: str, location: str, experience: str,
#               must_have: str, nice_to_have: str, notes: str) -> str:
#     llm = _get_llm()
#     user_msg = DRAFT_TEMPLATE.format(
#         role=role.strip(),
#         location=location.strip(),
#         experience=experience.strip(),
#         must_have=must_have.strip(),
#         nice_to_have=nice_to_have.strip(),
#         notes=notes.strip(),
#     )
#     resp = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT},
#                        {"role": "user", "content": user_msg}])
#     return (resp.content or "").strip()

# def _refine_jd(current_jd: str, feedback: str) -> str:
#     llm = _get_llm()
#     user_msg = REFINE_TEMPLATE.format(current_jd=current_jd.strip(), feedback=feedback.strip())
#     resp = llm.invoke([{"role": "system", "content": SYSTEM_PROMPT},
#                        {"role": "user", "content": user_msg}])
#     return (resp.content or "").strip()

# def create_jd_with_streamlit(jd_path: str = "job_description.txt") -> Optional[str]:
#     """
#     Streamlit UI for creating/refining a JD. Returns the saved JD text when saved,
#     otherwise None. Also writes the JD to `jd_path`.
#     """
#     st.header("✍️ Create a Job Description")

#     with st.expander("Inputs", expanded=True):
#         cols = st.columns(2)
#         with cols[0]:
#             role = st.text_input("Role/Title*", value=st.session_state.get("__jd_role__", "Software Engineer"))
#             location = st.text_input("Location*", value=st.session_state.get("__jd_location__", "Bengaluru (Hybrid)"))
#             experience = st.text_input("Experience (e.g., 3–6 years)*",
#                                        value=st.session_state.get("__jd_exp__", "3–6 years"))
#         with cols[1]:
#             must_have = st.text_area("Must-have skills* (comma-separated)",
#                                      value=st.session_state.get("__jd_must__", "Python, FastAPI, SQL, Docker"),
#                                      height=90)
#             nice_to_have = st.text_area("Good-to-have skills (comma-separated)",
#                                         value=st.session_state.get("__jd_nice__", "AWS, Terraform, LangChain, React"),
#                                         height=90)
#         notes = st.text_area("Additional notes (optional)", value=st.session_state.get("__jd_notes__", ""))

#         # Persist in session for back/forward
#         st.session_state["__jd_role__"] = role
#         st.session_state["__jd_location__"] = location
#         st.session_state["__jd_exp__"] = experience
#         st.session_state["__jd_must__"] = must_have
#         st.session_state["__jd_nice__"] = nice_to_have
#         st.session_state["__jd_notes__"] = notes

#         draft_clicked = st.button("🪄 Draft JD", type="primary")

#     if draft_clicked:
#         if not role.strip() or not location.strip() or not experience.strip() or not must_have.strip():
#             st.error("Please fill Role, Location, Experience, and Must-have skills.")
#         else:
#             with st.spinner("Drafting JD..."):
#                 st.session_state["__jd_draft__"] = _draft_jd(role, location, experience, must_have, nice_to_have, notes)

#     current = st.session_state.get("__jd_draft__", "")
#     if current:
#         st.subheader("Draft (editable)")
#         st.session_state["__jd_draft__"] = st.text_area("Job Description (Markdown)",
#                                                          value=current, height=420, key="__jd_draft_area__")

#         fb_cols = st.columns([3, 1])
#         with fb_cols[0]:
#             feedback = st.text_input("Refine with feedback (optional)",
#                                      value=st.session_state.get("__jd_feedback__", ""))
#             st.session_state["__jd_feedback__"] = feedback
#         with fb_cols[1]:
#             if st.button("↻ Refine"):
#                 if feedback.strip():
#                     with st.spinner("Refining..."):
#                         st.session_state["__jd_draft__"] = _refine_jd(st.session_state["__jd_draft__"], feedback)
#                 else:
#                     st.warning("Add some feedback first.")

#         save_cols = st.columns([1, 1, 2])
#         saved_text = None
#         with save_cols[0]:
#             if st.button("💾 Save JD"):
#                 text = (st.session_state.get("__jd_draft__") or "").strip()
#                 if not text:
#                     st.error("Nothing to save.")
#                 else:
#                     try:
#                         with open(jd_path, "w", encoding="utf-8") as f:
#                             f.write(text)
#                         st.success(f"Saved to {jd_path}")
#                         st.session_state["__jd_saved_now__"] = True
#                         saved_text = text
#                     except Exception as e:
#                         st.error(f"Failed to save: {e}")

#         with save_cols[1]:
#             if st.button("Use & Return →"):
#                 # This button is useful right after saving, to jump back to main flow
#                 text = (st.session_state.get("__jd_draft__") or "").strip()
#                 if text:
#                     try:
#                         with open(jd_path, "w", encoding="utf-8") as f:
#                             f.write(text)
#                         st.session_state["__jd_saved_now__"] = True
#                         saved_text = text
#                     except Exception as e:
#                         st.error(f"Failed to save: {e}")
#                 else:
#                     st.warning("Draft is empty. Save a draft first.")

#         # If saved this turn, return the text to caller so they can load it right away
#         if st.session_state.get("__jd_saved_now__") and saved_text:
#             return saved_text

#     st.caption("Uses OPENAI_API_KEY from your environment (.env). Model: " + DEFAULT_MODEL)
#     return None
# jd_agent.py
import os
from typing import Optional, Dict, Any, List

import streamlit as st
from dotenv import load_dotenv

# ---- LangChain / LangGraph imports ----
try:
    from langchain_openai import ChatOpenAI
except Exception:
    # Fallback for older LangChain installs
    from langchain.chat_models import ChatOpenAI  # type: ignore

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import MessagesState


# --------- Setup & constants ----------
load_dotenv()
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a concise technical recruiter & hiring manager assistant.
You help users draft and refine a job description (JD) from structured inputs.

Output should be markdown with these sections:
- Role summary (3–5 lines)
- Responsibilities (5–8 bullets)
- Must-have skills (bulleted)
- Good-to-have skills (bulleted)
- Location & work type
- Experience range
- Application/next steps

TOOLS:
- Use the `save_jd` tool to persist the final JD to disk when the user asks to save.
- When asked to save, ALWAYS call the tool. Do not simulate saving.
"""

DRAFT_TEMPLATE = """Create a job description with:

Role: {role}
Location: {location}
Experience: {experience}
Must-have skills (comma-separated): {must_have}
Good-to-have skills (comma-separated): {nice_to_have}
Additional notes: {notes}

Return the full JD in markdown with the required sections.
"""

REFINE_TEMPLATE = """Here is the current JD draft:

---
{current_jd}
---

Please revise according to this feedback:
{feedback}

Return ONLY the updated JD in markdown (no extra commentary).
"""


# --------- Tools (executed by ToolNode) ----------
@tool("save_jd")
def save_jd_tool(path: str, text: str) -> str:
    """Save a job description to disk at `path` with content `text` (UTF-8). Returns a confirmation string."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return f"JD saved to {path}"
    except Exception as e:
        return f"Failed to save JD to {path}: {e}"


TOOLS = [save_jd_tool]


# --------- Build the LangGraph app ----------
def _build_graph():
    """
    A minimal assistant->tools loop:
    - assistant node: ChatOpenAI bound to tools
    - tools node: ToolNode executes any tool calls
    - if the assistant returns tool calls -> go to tools; else -> END
    """
    graph = StateGraph(MessagesState)

    # Assistant node
    def call_llm(state: MessagesState) -> Dict[str, Any]:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.4)
        llm = llm.bind_tools(TOOLS)  # enable tool calling
        # Call the model on the rolling message history in state["messages"]
        resp = llm.invoke(state["messages"])
        # Append assistant message to the conversation
        return {"messages": [resp]}

    # Tool node (prebuilt)
    tools_node = ToolNode(TOOLS)

    graph.add_node("assistant", call_llm)
    graph.add_node("tools", tools_node)

    # Start at assistant
    graph.set_entry_point("assistant")

    # Conditional: if assistant called tools -> go to tools, else END.
    def route_after_assistant(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph.add_conditional_edges("assistant", route_after_assistant)
    # After tools, go back to assistant (to let the assistant summarize/confirm)
    graph.add_edge("tools", "assistant")

    app = graph.compile()
    return app


_APP = None
def _get_app():
    global _APP
    if _APP is None:
        _APP = _build_graph()
    return _APP


# --------- Small helpers that use the graph ----------
def _run_agent(messages: List[Any]) -> List[Any]:
    """
    Run one full assistant->(tools?)->assistant turn.
    Returns the updated message list (appended with assistant/tool messages).
    """
    app = _get_app()
    # The MessagesState expects a "messages" list; we pass that and take the output state's messages
    state = {"messages": messages}
    out = app.invoke(state)
    return out["messages"]


def _draft(role: str, location: str, experience: str,
           must_have: str, nice_to_have: str, notes: str) -> str:
    messages: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=DRAFT_TEMPLATE.format(
            role=role.strip(),
            location=location.strip(),
            experience=experience.strip(),
            must_have=must_have.strip(),
            nice_to_have=nice_to_have.strip(),
            notes=notes.strip(),
        ))
    ]
    messages = _run_agent(messages)
    # Last assistant content is the draft
    last = messages[-1]
    return getattr(last, "content", "").strip()


def _refine(current_jd: str, feedback: str) -> str:
    messages: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=REFINE_TEMPLATE.format(
            current_jd=current_jd.strip(),
            feedback=feedback.strip()
        ))
    ]
    messages = _run_agent(messages)
    last = messages[-1]
    return getattr(last, "content", "").strip()


def _save_via_tool(jd_text: str, jd_path: str) -> str:
    """
    Ask the assistant to call the save_jd tool.
    ToolNode will execute it and the assistant will confirm.
    Returns the assistant's final confirmation text.
    """
    messages: List[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        # Provide the complete JD and explicit save request
        HumanMessage(content=f"""Save the following JD to path "{jd_path}".
Use the save_jd tool (do not simulate).
JD:
\"\"\"{jd_text}\"\"\"""")
    ]
    messages = _run_agent(messages)
    # If the assistant called the tool, there will be a ToolMessage and a final AIMessage
    # We surface the final assistant text
    final_texts = [m.content for m in messages if isinstance(m, AIMessage)]
    return (final_texts[-1] if final_texts else "Done.")


# --------- Streamlit UI (kept identical signature) ----------
def create_jd_with_streamlit(jd_path: str = "job_description.txt") -> Optional[str]:
    """
    Streamlit UI for creating/refining a JD using a LangGraph+ToolNode agent.
    Returns the saved JD text when saved this turn (so caller can auto-load),
    otherwise None. Also writes the JD to `jd_path` via the save_jd tool.
    """
    st.header("✍️ Create a Job Description")

    with st.expander("Inputs", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            role = st.text_input("Role/Title*", value=st.session_state.get("__jd_role__", "Software Engineer"))
            location = st.text_input("Location*", value=st.session_state.get("__jd_location__", "Bengaluru (Hybrid)"))
            experience = st.text_input("Experience (e.g., 3–6 years)*",
                                       value=st.session_state.get("__jd_exp__", "3–6 years"))
        with cols[1]:
            must_have = st.text_area("Must-have skills* (comma-separated)",
                                     value=st.session_state.get("__jd_must__", "Python, FastAPI, SQL, Docker"),
                                     height=90)
            nice_to_have = st.text_area("Good-to-have skills (comma-separated)",
                                        value=st.session_state.get("__jd_nice__", "AWS, Terraform, LangChain, React"),
                                        height=90)
        notes = st.text_area("Additional notes (optional)", value=st.session_state.get("__jd_notes__", ""))

        # Persist to session for navigation comfort
        st.session_state["__jd_role__"] = role
        st.session_state["__jd_location__"] = location
        st.session_state["__jd_exp__"] = experience
        st.session_state["__jd_must__"] = must_have
        st.session_state["__jd_nice__"] = nice_to_have
        st.session_state["__jd_notes__"] = notes

        draft_clicked = st.button("🪄 Draft JD", type="primary")

    if draft_clicked:
        if not role.strip() or not location.strip() or not experience.strip() or not must_have.strip():
            st.error("Please fill Role, Location, Experience, and Must-have skills.")
        else:
            with st.spinner("Drafting JD..."):
                st.session_state["__jd_draft__"] = _draft(role, location, experience, must_have, nice_to_have, notes)

    current = st.session_state.get("__jd_draft__", "")
    if current:
        st.subheader("Draft (editable)")
        st.session_state["__jd_draft__"] = st.text_area("Job Description (Markdown)",
                                                         value=current, height=420, key="__jd_draft_area__")

        fb_cols = st.columns([3, 1])
        with fb_cols[0]:
            feedback = st.text_input("Refine with feedback (optional)",
                                     value=st.session_state.get("__jd_feedback__", ""))
            st.session_state["__jd_feedback__"] = feedback
        with fb_cols[1]:
            if st.button("↻ Refine"):
                if feedback.strip():
                    with st.spinner("Refining..."):
                        st.session_state["__jd_draft__"] = _refine(st.session_state["__jd_draft__"], feedback)
                else:
                    st.warning("Add some feedback first.")

        save_cols = st.columns([1, 1, 2])
        saved_text = None
        with save_cols[0]:
            if st.button("💾 Save JD"):
                text = (st.session_state.get("__jd_draft__") or "").strip()
                if not text:
                    st.error("Nothing to save.")
                else:
                    with st.spinner("Saving via tool..."):
                        confirmation = _save_via_tool(text, jd_path)
                    if "saved to" in confirmation.lower():
                        st.success(confirmation)
                        st.session_state["__jd_saved_now__"] = True
                        saved_text = text
                    else:
                        st.warning(confirmation or "Save attempt finished (see logs).")

        with save_cols[1]:
            if st.button("Use & Return →"):
                text = (st.session_state.get("__jd_draft__") or "").strip()
                if text:
                    with st.spinner("Saving via tool..."):
                        confirmation = _save_via_tool(text, jd_path)
                    if "saved to" in confirmation.lower():
                        st.session_state["__jd_saved_now__"] = True
                        saved_text = text
                        st.success(confirmation)
                    else:
                        st.warning(confirmation or "Save attempt finished (see logs).")
                else:
                    st.warning("Draft is empty. Save a draft first.")

        # If saved this turn, return the text so caller can auto-load and go back
        if st.session_state.get("__jd_saved_now__") and saved_text:
            return saved_text

    st.caption("Uses OPENAI_API_KEY from your environment (.env). Model: " + DEFAULT_MODEL)
    return None

