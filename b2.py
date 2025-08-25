# main.py
import io
import os
from datetime import datetime
import json
import glob
import pandas as pd
import streamlit as st

from agent1 import shortlist_with_streamlit_inputs
from scheduler_agent import run_schedule, accept_suggestion, load_interview_schedule

st.set_page_config(page_title="Shortlist & Schedule", page_icon="✅📅", layout="wide")

DOCS_DIR = "./docs"
JD_PATH = "job_description.txt"

# ---------------- Session State (robust to reruns) ----------------
def _init_state():
    ss = st.session_state
    ss.setdefault("stage", "shortlist")  # "shortlist" | "schedule" | "end"

    # JD state
    ss.setdefault("jd_mode", "Load from job_description.txt")
    ss.setdefault("jd_text", "")
    ss.setdefault("__jd_pdf__", None)            # raw bytes if user uploads a JD PDF
    ss.setdefault("__jd_loaded_from_disk__", False)

    # Resumes state
    ss.setdefault("resume_mode", "Load from ./docs")  # NEW
    ss.setdefault("resume_files_cache", [])     # list[(name, bytes)]
    ss.setdefault("__docs_scan__", [])          # list of file paths in ./docs
    ss.setdefault("__docs_loaded__", set())     # track which docs were loaded (filenames)

    # Shortlisting/scheduler state
    ss.setdefault("top_n", 2)
    ss.setdefault("shortlisted", [])            # list[dict] from agent (top N)
    ss.setdefault("scores_df", pd.DataFrame())
    ss.setdefault("last_exception", None)
    ss.setdefault("schedule_df", load_interview_schedule())
    ss.setdefault("last_result", None)
    ss.setdefault("candidate", "")
    ss.setdefault("raw_time", "")

_init_state()

# ---------- Helpers ----------
def _read_job_desc_from_disk(path: str = JD_PATH) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _scan_docs(dirpath: str = DOCS_DIR):
    patterns = ["*.pdf", "*.txt"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dirpath, pat)))
    files = sorted(files)
    return files

def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

st.title("✅ Shortlist Candidates → 📅 Schedule Interviews")
st.caption("Give me a Job Description + resumes; I’ll rank them, then (optionally) schedule interviews—maintaining context the whole way.")

# ============ STAGE: SHORTLIST ============
if st.session_state.stage == "shortlist":
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            - Extracts text from each resume (PDF/TXT)
            - Builds a weighted scoring rubric from the JD (LangGraph + LLM)
            - Scores all resumes, ranks them, and returns your **Top N**
            - Then you can jump straight into scheduling interviews
            """
        )

    # --- JD Input ---
    st.subheader("1) Job Description")
    st.session_state.jd_mode = st.radio(
        "Provide JD by:",
        ["Load from job_description.txt", "Upload file"],
        horizontal=True,
        index=0 if st.session_state.jd_mode == "Load from job_description.txt" else 1
    )

    if st.session_state.jd_mode == "Load from job_description.txt":
        if os.path.exists(JD_PATH):
            mtime = datetime.fromtimestamp(os.path.getmtime(JD_PATH)).strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"Found **{JD_PATH}** (last modified: {mtime}).")

            colA, colB = st.columns([1, 3])
            with colA:
                if st.button("Reload JD from disk"):
                    st.session_state.jd_text = _read_job_desc_from_disk(JD_PATH)
                    st.session_state.__jd_loaded_from_disk__ = True

            if not st.session_state.__jd_loaded_from_disk__:
                st.session_state.jd_text = _read_job_desc_from_disk(JD_PATH)
                st.session_state.__jd_loaded_from_disk__ = True

            st.text_area("Preview (read-only)", value=st.session_state.jd_text, height=220, disabled=True)
            st.session_state.__jd_pdf__ = None
        else:
            st.warning("No **job_description.txt** found in the working directory. Switch to **Upload file** to provide a JD.")
            st.session_state.jd_text = ""
            st.session_state.__jd_pdf__ = None
    else:
        jd_file = st.file_uploader("Upload JD (.pdf or .txt)", type=["pdf", "txt"], key="jd_uploader")
        if jd_file is not None:
            name = jd_file.name.lower()
            if name.endswith(".txt"):
                st.session_state.jd_text = jd_file.read().decode("utf-8", errors="ignore")
                st.session_state.__jd_pdf__ = None
                st.session_state.__jd_loaded_from_disk__ = False
                st.success("Loaded JD from uploaded .txt")
            else:
                st.session_state.__jd_pdf__ = jd_file.read()
                st.session_state.jd_text = ""
                st.session_state.__jd_loaded_from_disk__ = False
                st.info("JD stored as PDF bytes; your agent may parse it.")

    # --- Resumes Source (NEW) ---
    st.subheader("2) Resumes")
    st.session_state.resume_mode = st.radio(
        "Get resumes by:",
        ["Load from ./docs", "Upload files"],
        horizontal=True,
        index=0 if st.session_state.resume_mode == "Load from ./docs" else 1,
        key="resume_mode_radio"
    )

    if st.session_state.resume_mode == "Load from ./docs":
        # Scan ./docs
        cols = st.columns([1, 1, 2])
        with cols[0]:
            if st.button("Refresh ./docs"):
                st.session_state.__docs_scan__ = _scan_docs(DOCS_DIR)

        # Auto-scan first time
        if not st.session_state.__docs_scan__:
            st.session_state.__docs_scan__ = _scan_docs(DOCS_DIR)

        files = st.session_state.__docs_scan__
        if not files:
            st.warning(f"No .pdf or .txt files found in **{DOCS_DIR}**. Add resumes there or switch to **Upload files**.")
            st.session_state.resume_files_cache = []
        else:
            pretty_names = [os.path.basename(p) for p in files]
            default_selection = pretty_names  # preselect all
            picked = st.multiselect(
                "Select resumes from ./docs",
                options=pretty_names,
                default=default_selection,
                key="docs_multiselect"
            )

            with cols[1]:
                if st.button("Load selected"):
                    loaded = []
                    for name in picked:
                        path = os.path.join(DOCS_DIR, name)
                        try:
                            b = _read_file_bytes(path)
                            loaded.append((name, b))
                        except Exception as e:
                            st.error(f"Failed to read {name}: {e}")
                    st.session_state.resume_files_cache = loaded
                    st.session_state.__docs_loaded__ = set(picked)
                    st.success(f"Loaded {len(loaded)} resume(s) from ./docs")

            # Small preview table
            if st.session_state.resume_files_cache:
                df_prev = pd.DataFrame(
                    {"file": [n for n, _ in st.session_state.resume_files_cache],
                     "size_kb": [round(len(b)/1024, 1) for _, b in st.session_state.resume_files_cache]}
                )
                st.dataframe(df_prev, use_container_width=True, hide_index=True)

    else:
        resume_files = st.file_uploader(
            "Upload multiple resumes (.pdf or .txt)", type=["pdf", "txt"], accept_multiple_files=True
        )
        if resume_files:
            st.session_state.resume_files_cache = [(f.name, f.read()) for f in resume_files]
            df_prev = pd.DataFrame(
                {"file": [n for n, _ in st.session_state.resume_files_cache],
                 "size_kb": [round(len(b)/1024, 1) for _, b in st.session_state.resume_files_cache]}
            )
            st.dataframe(df_prev, use_container_width=True, hide_index=True)

    # --- Shortlisting Options ---
    st.subheader("3) Shortlisting Options")
    st.session_state.top_n = st.number_input(
        "How many candidates do you want to shortlist?",
        min_value=1, max_value=10, value=st.session_state.top_n
    )

    run = st.button("Run Shortlisting", type="primary")

    def _validate_inputs():
        # JD presence
        if st.session_state.jd_mode == "Load from job_description.txt":
            if not st.session_state.jd_text.strip() and not st.session_state.__jd_pdf__:
                st.error("Could not load JD from job_description.txt. Ensure the file exists or switch to Upload.")
                return False
        else:
            if (not st.session_state.jd_text.strip()) and (st.session_state.__jd_pdf__ is None):
                st.error("Please upload a JD (.txt preferred, .pdf if your agent handles it).")
                return False

        # Resumes presence
        if not st.session_state.resume_files_cache:
            st.error("Please load/upload at least one resume.")
            return False
        return True

    if run and _validate_inputs():
        try:
            top, scores = shortlist_with_streamlit_inputs(
                jd_text=st.session_state.jd_text,
                uploaded_files=st.session_state.resume_files_cache,
                top_n=int(st.session_state.top_n),
            )
            st.session_state.shortlisted = top or []
            st.session_state.scores_df = pd.DataFrame(scores or [])
            st.session_state.last_exception = None
        except Exception as e:
            st.session_state.shortlisted = []
            st.session_state.scores_df = pd.DataFrame()
            st.session_state.last_exception = e

    # ----- Results Area -----
    if st.session_state.last_exception:
        st.exception(st.session_state.last_exception)

    if len(st.session_state.shortlisted) > 0:
        st.markdown("---")
        st.subheader(f"Top {st.session_state.top_n} Candidates")
        for i, row in enumerate(st.session_state.shortlisted, 1):
            with st.container(border=True):
                st.markdown(f"### #{i} — {row.get('candidate_file','(unknown)')}")
                total = row.get("total_score")
                if total is not None:
                    try:
                        st.metric("Total Score", f"{float(total):.2f}")
                    except Exception:
                        st.metric("Total Score", str(total))
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Strengths**")
                    for s in row.get("strengths", []) or []:
                        st.write(f"- {s}")
                with cols[1]:
                    st.markdown("**Concerns**")
                    for c in row.get("concerns", []) or []:
                        st.write(f"- {c}")
                with st.expander("Rationale"):
                    st.write(row.get("rationale", ""))

    # Full ranking table + downloads
    if not st.session_state.scores_df.empty:
        st.subheader("Full Ranking")
        df = st.session_state.scores_df.copy()
        if "total_score" in df.columns:
            df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").round(2)
        st.dataframe(df.sort_values("total_score", ascending=False), use_container_width=True, hide_index=True)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scores.csv",
            data=csv_bytes,
            file_name="scores.csv",
            mime="text/csv",
        )

        shortlist_json_bytes = json.dumps(st.session_state.shortlisted, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Download shortlist.json",
            data=shortlist_json_bytes,
            file_name="shortlist.json",
            mime="application/json",
        )

    # ------ Next Step ------
    if len(st.session_state.shortlisted) > 0:
        st.markdown("---")
        st.subheader("Next Step")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Schedule interviews →", key="go_schedule"):
                first_name = (st.session_state.shortlisted[0].get("candidate_file", "").rsplit(".", 1)[0])
                if first_name:
                    st.session_state.candidate = first_name
                st.session_state.stage = "schedule"
                st.session_state.pop("next_action_choice", None)
                st.rerun()

        with c2:
            if st.button("END", key="end_flow"):
                st.session_state.stage = "end"
                st.session_state.pop("next_action_choice", None)
                st.rerun()

# ============ STAGE: SCHEDULE ============
elif st.session_state.stage == "schedule":
    left, right = st.columns([3, 2])
    with left:
        st.header("📅 Interview Scheduler")
        with st.expander("Current Schedule", expanded=True):
            st.dataframe(st.session_state.schedule_df, use_container_width=True)
            buff = io.StringIO()
            st.session_state.schedule_df.to_csv(buff, index=False)
            st.download_button(
                "Download CSV",
                data=buff.getvalue(),
                file_name="job_interview.csv",
                mime="text/csv",
            )

        st.subheader("Schedule a Candidate")

        shortlist_names = [
            (c.get("candidate_file", "") or "").rsplit(".", 1)[0] for c in st.session_state.shortlisted
        ]
        shortlist_names = [n for n in shortlist_names if n]

        if shortlist_names:
            picked = st.selectbox(
                "Pick from shortlisted (or choose 'Custom' to type):",
                options=["Custom"] + shortlist_names,
                index=0,
                key="picked_shortlist_name"
            )
            if picked != "Custom":
                st.session_state.candidate = picked

        st.session_state.candidate = st.text_input(
            "Candidate name",
            value=st.session_state.candidate,
            placeholder="Jane Doe",
            key="candidate_input"
        )
        st.session_state.raw_time = st.text_input(
            "Preferred time (e.g., 15:00 or 3pm)",
            value=st.session_state.raw_time,
            key="time_input"
        )

        colA, colB = st.columns([1,1])
        with colA:
            submitted = st.button("Schedule", type="primary")
        with colB:
            if st.button("← Back to Shortlisting", key="back_btn"):
                st.session_state.stage = "shortlist"
                st.session_state.last_result = None
                st.session_state.pop("next_action_choice", None)
                st.rerun()

        if submitted:
            result = run_schedule(st.session_state.candidate.strip(), st.session_state.raw_time.strip())
            st.session_state.last_result = result
            st.session_state.schedule_df = load_interview_schedule()

        # Results
        if st.session_state.last_result:
            r = st.session_state.last_result
            status = r.get("status")
            if status == "invalid_time":
                st.error(r.get("message", "Invalid time"))
            elif status == "no_slots":
                st.error("No slots available today. Try another day.")
            elif status == "suggested":
                st.warning(r.get("message", "Requested slot is taken."))
                suggested = r.get("suggested_time")
                if suggested:
                    if st.button(f"Accept suggested slot {suggested}", key=f"accept_{suggested}"):
                        accept_result = accept_suggestion(st.session_state.candidate, suggested)
                        st.session_state.last_result = accept_result
                        st.success(accept_result.get("message", "Scheduled."))
                        st.session_state.schedule_df = load_interview_schedule()
            elif status == "scheduled":
                st.success(r.get("message", "Scheduled."))
            else:
                st.info("Ready.")

    with right:
        st.markdown("### Shortlisted (Context)")
        if st.session_state.shortlisted:
            view_df = pd.DataFrame(st.session_state.shortlisted)
            cols = [c for c in ["candidate_file", "total_score"] if c in view_df.columns]
            if cols:
                small = view_df[cols].copy()
                if "total_score" in small.columns:
                    small["total_score"] = pd.to_numeric(small["total_score"], errors="coerce").round(2)
                st.dataframe(small.sort_values("total_score", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No shortlisted candidates yet. Go back and run shortlisting first.")

# ============ STAGE: END ============
elif st.session_state.stage == "end":
    st.header("All done 🎉")
    st.write("You chose to END. You can restart or jump to scheduling anytime.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Restart Flow"):
            st.session_state.stage = "shortlist"
            st.session_state.shortlisted = []
            st.session_state.scores_df = pd.DataFrame()
            st.session_state.jd_text = ""
            st.session_state.__jd_pdf__ = None
            st.session_state.resume_files_cache = []
            st.session_state.last_exception = None
            st.session_state.top_n = 2
            st.session_state.__jd_loaded_from_disk__ = False
            st.session_state.__docs_scan__ = []
            st.session_state.__docs_loaded__ = set()
            st.rerun()
    with c2:
        if st.button("Go to Scheduler"):
            st.session_state.stage = "schedule"
            st.rerun()

st.markdown("---")
st.caption(
    "Tip: Keep **job_description.txt** in the app folder for one-click loading. "
    "Place resumes in **./docs** or upload them here. Resumes: **PDF/TXT**. "
    "Times allowed: 12:00–18:00 (60-minute steps)."
)
