# app.py
# Master Resume Agent — LangGraph × Streamlit (3 modes fully wired)
# Fixes:
#  - JD: review screen with feedback + revise/confirm actions (immediate navigation after generate)
#  - Shortlisting: explicit "Evaluate Resume" button right after upload
#
# Prereqs:
#   pip install streamlit langgraph langchain-openai langchain-community python-dotenv pypdf pandas
#   export OPENAI_API_KEY=...  (or set in a .env file)

import os
import json
import pandas as pd
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate

# -----------------------
# ENV
# -----------------------
load_dotenv()  # picks up OPENAI_API_KEY if present in .env

# -----------------------
# GRAPH STATE
# -----------------------
class GraphState(dict):
    mode: str
    job_title: str
    must_have: list
    nice_to_have: list
    responsibilities: list
    location: str
    experience: str
    job_description: str
    resume_text: str
    extra_criteria: list
    evaluation: str

# -----------------------
# CONSTANTS / FILES
# -----------------------
CRITERIA_FILE = "criteria.json"
JD_FILE = "job_description.txt"
INTERVIEW_FILE = "job_interview.csv"

# -----------------------
# UTILITIES
# -----------------------
def save_criteria(must_have, nice_to_have, file_path=CRITERIA_FILE):
    with open(file_path, "w") as f:
        json.dump({"must_have": must_have, "nice_to_have": nice_to_have}, f, indent=2)

def save_job_description(jd_text, file_path=JD_FILE):
    with open(file_path, "w") as f:
        f.write(jd_text)

def load_criteria(file_path=CRITERIA_FILE):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["must_have"], data["nice_to_have"]

def load_interview_schedule():
    try:
        return pd.read_csv(INTERVIEW_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["candidate_name", "interview_time"])

def save_interview_schedule(df):
    df.to_csv(INTERVIEW_FILE, index=False)

def goto(node: str):
    """Hard navigate to a node and rerun for immediate UI update."""
    st.session_state.current_node = node
    st.rerun()

# -----------------------
# LLM HELPERS
# -----------------------
def normalize_time_with_llm(raw_time: str) -> str:
    """Return HH:MM (24h) between 12:00–18:00 or 'INVALID'."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["raw_time"],
        template="""
You are a helpful assistant. Convert the following human-provided interview time into
24-hour HH:MM format (e.g., 12:00, 15:00, 18:00).  

Rules:
- Only output the normalized time (nothing else).
- Valid range is 12:00 to 18:00.  
- If input is outside this range, return "INVALID".

Input: {raw_time}
Output:
""",
    )
    chain = prompt | llm
    return chain.invoke({"raw_time": raw_time}).content.strip()

def suggest_alternative_with_llm(requested_time, available_slots):
    """Pick closest available slot to the requested_time; return HH:MM only."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = PromptTemplate(
        input_variables=["requested_time", "slots"],
        template="""
A recruiter requested an interview at {requested_time}, but that slot is unavailable.
Here are the available slots: {slots}

Suggest the single best alternative (closest to requested time). 
Return ONLY the time in HH:MM format (24h).
""",
    )
    chain = prompt | llm
    return chain.invoke(
        {"requested_time": requested_time, "slots": ", ".join(available_slots)}
    ).content.strip()

# -----------------------
# LANGGRAPH NODES (Streamlit UI)
# -----------------------

def choose_mode_node(state: GraphState):
    st.header("Streamline Recruitment with AI Agents")
    st.caption("Pick a workflow to run.")
    with st.form("choose_mode_form", clear_on_submit=False):
        choice = st.radio(
            "Choose mode",
            ["Create Job Description & Criteria", "Shortlist Candidate Resume", "Schedule Interview"],
            index={"jd":0, "shortlist":1, "interview":2}.get(state.get("mode", "jd"), 0),
        )
        submitted = st.form_submit_button("Continue")

    if submitted:
        if choice.startswith("Create"):
            state.update({"mode": "jd"})
            goto("create_jd")
        elif choice.startswith("Shortlist"):
            state.update({"mode": "shortlist"})
            goto("load_resume")
        else:
            state.update({"mode": "interview"})
            goto("interview_scheduler")
    return {}

def create_jd_node(state: GraphState):
    st.subheader("Create Job Description")
    with st.form("create_jd_form"):
        job_title = st.text_input("Job Title", state.get("job_title", ""))
        responsibilities = st.text_area(
            "Responsibilities (one per line)",
            "\n".join(state.get("responsibilities", [])),
            height=150,
        )
        must_have = st.text_area(
            "Must-Have Skills (one per line)",
            "\n".join(state.get("must_have", [])),
            height=150,
        )
        nice_to_have = st.text_area(
            "Nice-to-Have Skills (one per line)",
            "\n".join(state.get("nice_to_have", [])),
            height=150,
        )
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location", state.get("location", ""))
        with col2:
            experience = st.text_input("Experience Requirement (e.g., 3-5 years)", state.get("experience", ""))

        submitted = st.form_submit_button("Generate JD")

    if submitted:
        responsibilities_list = [r.strip() for r in responsibilities.splitlines() if r.strip()]
        must_have_list = [m.strip() for m in must_have.splitlines() if m.strip()]
        nice_to_have_list = [n.strip() for n in nice_to_have.splitlines() if n.strip()]
        save_criteria(must_have_list, nice_to_have_list)
        updates = {
            "job_title": job_title,
            "responsibilities": responsibilities_list,
            "must_have": must_have_list,
            "nice_to_have": nice_to_have_list,
            "location": location,
            "experience": experience,
        }
        state.update(updates)
        goto("generate_jd")
    return {}

def generate_jd_node(state: GraphState):
    st.subheader("Generate JD with LLM")
    with st.spinner("Generating a professional job description..."):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        prompt = PromptTemplate(
            input_variables=["title", "responsibilities", "must_have", "nice_to_have", "location", "experience"],
            template="""
Write a professional job description for the following role.

Job Title: {title}
Location: {location}
Experience Required: {experience}

Responsibilities:
{responsibilities}

Must-Have Skills:
{must_have}

Nice-to-Have Skills:
{nice_to_have}

Format it clearly with sections:
- Job Title
- Location
- Experience Required
- Responsibilities
- Must-Have Skills
- Nice-to-Have Skills
""",
        )
        jd_text = (prompt | llm).invoke(
            {
                "title": state["job_title"],
                "responsibilities": "\n".join(state["responsibilities"]) if state.get("responsibilities") else "Not specified",
                "must_have": "\n".join(state["must_have"]) if state.get("must_have") else "Not specified",
                "nice_to_have": "\n".join(state["nice_to_have"]) if state.get("nice_to_have") else "Not specified",
                "location": state.get("location") or "Not specified",
                "experience": state.get("experience") or "Not specified",
            }
        ).content

    save_job_description(jd_text)
    state["job_description"] = jd_text
    # Immediately go to review screen with feedback controls
    goto("human_feedback_jd")
    return {}

def human_feedback_jd_node(state: GraphState):
    st.subheader("Review & Improve JD")
    st.text_area("Current JD", state.get("job_description", ""), height=350, key="jd_display")

    with st.form("jd_feedback_form"):
        feedback = st.text_area("Feedback to improve JD", "", height=150)
        col1, col2 = st.columns(2)
        with col1:
            confirm = st.form_submit_button("✅ Confirm JD")
        with col2:
            revise = st.form_submit_button("✏️ Revise with LLM")

    if confirm:
        st.success("Final JD confirmed. Saved to job_description.txt")
        return END

    if revise:
        if not feedback.strip():
            st.warning("Please enter feedback before revising.")
            return {}
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        prompt = PromptTemplate(
            input_variables=["jd", "feedback"],
            template="""
You are a hiring assistant. 
Here is the current job description:

{jd}

The recruiter provided the following feedback:
{feedback}

Revise the job description accordingly while keeping it professional and well-structured.
Return only the improved job description.
""",
        )
        new_jd = (prompt | llm).invoke({"jd": state["job_description"], "feedback": feedback}).content
        save_job_description(new_jd)
        state["job_description"] = new_jd
        st.success("JD updated & saved.")
        # Stay on this page for iterative edits
        goto("human_feedback_jd")
    return {}

def load_resume_node(state: GraphState):
    st.subheader("Upload Resume PDF")
    uploaded = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"])

    # Criteria handling
    st.caption("Shortlisting uses saved criteria from the JD flow. If missing, enter them below.")
    must_have, nice_to_have = [], []
    criteria_exists = os.path.exists(CRITERIA_FILE)

    with st.expander("Criteria (used for shortlisting)"):
        if criteria_exists:
            try:
                must_have, nice_to_have = load_criteria()
                st.write("Loaded saved criteria from criteria.json")
                st.write("**Must-Have:**")
                st.code("\n".join(must_have) or "—")
                st.write("**Nice-to-Have:**")
                st.code("\n".join(nice_to_have) or "—")
            except Exception:
                criteria_exists = False

        if not criteria_exists:
            mh_text = st.text_area("Must-Have (one per line)", "", height=120)
            nh_text = st.text_area("Nice-to-Have (one per line)", "", height=120)
            if st.button("Save Criteria"):
                must_have = [x.strip() for x in mh_text.splitlines() if x.strip()]
                nice_to_have = [x.strip() for x in nh_text.splitlines() if x.strip()]
                save_criteria(must_have, nice_to_have)
                st.success("Saved criteria.json")

    if uploaded:
        pdf_path = "uploaded_resume.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded.read())
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        resume_text = "\n".join([p.page_content for p in pages])

        # ensure criteria loaded (if exists)
        if os.path.exists(CRITERIA_FILE) and not (must_have and nice_to_have):
            must_have, nice_to_have = load_criteria()

        st.success("Resume uploaded. Click the button below to evaluate.")
        if st.button("🔍 Evaluate Resume"):
            state.update({
                "resume_text": resume_text,
                "must_have": must_have,
                "nice_to_have": nice_to_have,
                "extra_criteria": state.get("extra_criteria", []),
            })
            goto("evaluate_resume")
    return {}

def evaluate_resume_node(state: GraphState):
    st.subheader("Evaluate Resume")
    if not state.get("resume_text"):
        st.info("Upload a resume, then click 'Evaluate Resume'.")
        return {}

    extra_display = "\n".join(state.get("extra_criteria", [])) if state.get("extra_criteria") else "None"

    with st.spinner("Evaluating with LLM..."):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = PromptTemplate(
            input_variables=["resume_text", "must_have", "nice_to_have", "extra"],
            template="""
You are a hiring assistant.

Must-Have Criteria:
{must_have}

Nice-to-Have Criteria:
{nice_to_have}

Extra Criteria from Recruiter:
{extra}

Rules:
- Candidate must meet ALL must-have criteria to be shortlisted.
- Nice-to-have and extra criteria improve chances but are not mandatory.

Resume:
{resume_text}

Return:
Decision: <SHORTLIST or REJECT>
Reason:
- ...
- ...
""",
        )
        evaluation = (prompt | llm).invoke(
            {
                "resume_text": state["resume_text"],
                "must_have": "\n".join(state.get("must_have", [])),
                "nice_to_have": "\n".join(state.get("nice_to_have", [])),
                "extra": extra_display,
            }
        ).content

    state["evaluation"] = evaluation
    st.text_area("Evaluation Result", evaluation, height=280)
    goto("human_check")
    return {}

def human_check_node(state: GraphState):
    st.subheader("Add Extra Criteria or Finalize")
    st.text_area("Latest Evaluation", state.get("evaluation",""), height=220, disabled=True)
    with st.form("extra_form"):
        new_c = st.text_input("Add extra criteria (optional)")
        col1, col2 = st.columns(2)
        with col1:
            add = st.form_submit_button("➕ Add & Re-evaluate")
        with col2:
            finalize = st.form_submit_button("✅ Finalize")

    if add:
        if not new_c.strip():
            st.warning("Please enter a criterion to add.")
            return {}
        updated_extra = (state.get("extra_criteria") or []) + [new_c.strip()]
        state["extra_criteria"] = updated_extra
        goto("evaluate_resume")

    if finalize:
        st.success("Resume evaluation finalized.")
        return END

    return {}

def interview_scheduler_node(state: GraphState):
    st.subheader("Interview Scheduler")

    # current schedule
    df = load_interview_schedule()
    if df.empty:
        st.info("No interviews scheduled yet.")
    else:
        st.dataframe(df, use_container_width=True)

    # persistent UI state for suggestion flow
    if "pending_suggestion" not in st.session_state:
        st.session_state.pending_suggestion = None  # dict: {candidate, suggested}
    if "pending_time_conflict" not in st.session_state:
        st.session_state.pending_time_conflict = None  # str requested time

    with st.form("schedule_form"):
        candidate = st.text_input("Candidate name", value=state.get("candidate_name", ""))
        raw_time = st.text_input("Preferred interview time (e.g., 3pm, 3:00 pm, 15:00)", value=state.get("raw_time", ""))
        colA, colB = st.columns(2)
        with colA:
            exit_click = st.form_submit_button("🚪 Exit scheduler")
        with colB:
            schedule_click = st.form_submit_button("📅 Try to schedule")

    if exit_click:
        st.info("Exiting scheduler.")
        return END

    # Handle confirm/decline for suggested slot (if present)
    if st.session_state.pending_suggestion:
        sug = st.session_state.pending_suggestion
        st.warning(f"Slot {st.session_state.pending_time_conflict} is taken. Suggested alternative: {sug['suggested']}")
        colA, colB = st.columns(2)
        with colA:
            accept = st.button(f"✅ Schedule {sug['candidate']} at {sug['suggested']}")
        with colB:
            decline = st.button("❌ Decline suggestion")

        if accept:
            df2 = load_interview_schedule()
            df2 = pd.concat(
                [df2, pd.DataFrame([[sug["candidate"], sug["suggested"]]], columns=["candidate_name", "interview_time"])],
                ignore_index=True,
            )
            save_interview_schedule(df2)
            st.success(f"Scheduled {sug['candidate']} at {sug['suggested']}.")
            st.session_state.pending_suggestion = None
            st.session_state.pending_time_conflict = None
            return END

        if decline:
            st.info("Not scheduled. You can try another time.")
            st.session_state.pending_suggestion = None
            st.session_state.pending_time_conflict = None
            return {}

    if schedule_click:
        if not candidate.strip() or not raw_time.strip():
            st.error("Please provide both candidate name and time.")
            return {}

        preferred_time = normalize_time_with_llm(raw_time)
        if preferred_time == "INVALID":
            st.error("Invalid time. Interviews can only be scheduled between 12:00 and 18:00 (inclusive).")
            return {}

        df_now = load_interview_schedule()
        booked_times = df_now["interview_time"].tolist() if not df_now.empty else []

        if preferred_time in booked_times:
            st.warning(f"⚠️ Slot at {preferred_time} is already taken.")
            # build hourly slots between 12:00 and 18:00
            start = datetime.strptime("12:00", "%H:%M")
            end = datetime.strptime("18:00", "%H:%M")
            available_slots = []
            cur = start
            while cur <= end:
                slot = cur.strftime("%H:%M")
                if slot not in booked_times:
                    available_slots.append(slot)
                cur += timedelta(hours=1)

            if available_slots:
                suggested = suggest_alternative_with_llm(preferred_time, available_slots)
                st.session_state.pending_suggestion = {"candidate": candidate, "suggested": suggested}
                st.session_state.pending_time_conflict = preferred_time
                st.info("Review the suggestion above to confirm or decline.")
                return {}
            else:
                st.error("No slots available today.")
                return END
        else:
            df_new = pd.concat(
                [df_now, pd.DataFrame([[candidate, preferred_time]], columns=["candidate_name", "interview_time"])],
                ignore_index=True,
            )
            save_interview_schedule(df_new)
            st.success(f"✅ Scheduled {candidate} at {preferred_time}.")
            return END

    return {}

# -----------------------
# BUILD MASTER GRAPH
# -----------------------
workflow = StateGraph(GraphState)

workflow.add_node("choose_mode", choose_mode_node)

workflow.add_node("create_jd", create_jd_node)
workflow.add_node("generate_jd", generate_jd_node)
workflow.add_node("human_feedback_jd", human_feedback_jd_node)

workflow.add_node("load_resume", load_resume_node)
workflow.add_node("evaluate_resume", evaluate_resume_node)
workflow.add_node("human_check", human_check_node)

workflow.add_node("interview_scheduler", interview_scheduler_node)

workflow.set_entry_point("choose_mode")
workflow.add_conditional_edges(
    "choose_mode",
    lambda state: state["mode"],
    {"jd": "create_jd", "shortlist": "load_resume", "interview": "interview_scheduler"},
)
workflow.add_edge("create_jd", "generate_jd")
workflow.add_edge("generate_jd", "human_feedback_jd")
workflow.add_edge("human_feedback_jd", "human_feedback_jd")  # loop until confirmed

workflow.add_edge("load_resume", "evaluate_resume")
workflow.add_edge("evaluate_resume", "human_check")
workflow.add_edge("human_check", "evaluate_resume")  # loop when extra criteria added

# -----------------------
# NODE REGISTRY
# -----------------------
NODES = {
    "choose_mode": choose_mode_node,
    "create_jd": create_jd_node,
    "generate_jd": generate_jd_node,
    "human_feedback_jd": human_feedback_jd_node,
    "load_resume": load_resume_node,
    "evaluate_resume": evaluate_resume_node,
    "human_check": human_check_node,
    "interview_scheduler": interview_scheduler_node,
}

# -----------------------
# STREAMLIT DRIVER
# -----------------------
def step_driver(current_node: str, state: GraphState):
    """Render the current node."""
    node_fn = NODES.get(current_node)
    if node_fn is None:
        st.error(f"Unknown node: {current_node}")
        st.session_state.current_node = "choose_mode"
        return
    result = node_fn(state)

    if result is END:
        st.session_state.current_node = "choose_mode"
        return

    if isinstance(result, dict) and result:
        state.update(result)

def main():
    st.set_page_config(page_title="Hire Agent", page_icon="🧭", layout="wide")
    st.title("🤯 Hiring Agent — Powered by LangGraph")

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Reset App State"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.caption("Persists criteria (criteria.json), JD (job_description.txt), schedule (job_interview.csv) locally.")

    # Initialize graph state & current node
    if "graph_state" not in st.session_state:
        st.session_state.graph_state = GraphState()
    if "current_node" not in st.session_state:
        st.session_state.current_node = "choose_mode"

    # Drive current node
    step_driver(st.session_state.current_node, st.session_state.graph_state)

if __name__ == "__main__":
    main()
