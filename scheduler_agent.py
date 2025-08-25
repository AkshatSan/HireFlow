# `scheduler_agent.py` (LangGraph + Pydantic, robust time extraction)


from __future__ import annotations

import io
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, constr

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

load_dotenv()

# -------------------- Storage --------------------
INTERVIEW_FILE = os.getenv("INTERVIEW_FILE", "job_interview.csv")
TIME_START = "12:00"
TIME_END = "18:00"
STEP_MINUTES = 60


def load_interview_schedule(file_path: str = INTERVIEW_FILE) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["candidate_name", "interview_time"])
    # enforce schema/order
    for col in ["candidate_name", "interview_time"]:
        if col not in df.columns:
            df[col] = ""
    return df[["candidate_name", "interview_time"]]


def save_interview_schedule(df: pd.DataFrame, file_path: str = INTERVIEW_FILE) -> None:
    df = df[["candidate_name", "interview_time"]]
    df.to_csv(file_path, index=False)


# -------------------- Pydantic Schemas --------------------
class GraphState(BaseModel):
    candidate: str = ""
    raw_time: str = ""
    preferred_time: Optional[str] = None
    status: Literal[
        "idle", "invalid_time", "suggested", "no_slots", "scheduled"
    ] = "idle"
    message: str = ""
    suggested_time: Optional[str] = None
    available_slots: List[str] = Field(default_factory=list)
    schedule_df_csv: Optional[str] = None  # UI convenience: df.to_csv string


class NormalizedTime(BaseModel):
    status: Literal["ok", "invalid"]
    time: Optional[str] = Field(
        default=None,
        pattern=r'^(?:[01]\d|2[0-3]):[0-5]\d$'
    )


class SuggestedTime(BaseModel):
    suggestion: Optional[str] = Field(
        default=None,
        pattern=r'^(?:[01]\d|2[0-3]):[0-5]\d$'
    )

# -------------------- LLM helpers with Pydantic parsers --------------------
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _preextract_time_token(text: str) -> Optional[str]:
    """Heuristic extractor to pull a time token from arbitrary text.
    Returns a 24h HH:MM candidate if found, else None.
    """
    if not text:
        return None
    t = text.lower()
    # e.g., "at 2pm", "for 2:30 pm", "by 14:00", "around 4 pm"
    # 1) with am/pm
    m = re.search(r"\b(?:at|for|by|around|about|near|~)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2)) if m.group(2) else 0
        suffix = m.group(3)
        if suffix == "pm" and hh != 12:
            hh += 12
        if suffix == "am" and hh == 12:
            hh = 0
        return f"{hh:02d}:{mm:02d}"
    # 2) 24h like 14:00
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        return f"{hh:02d}:{mm:02d}"
    return None


def normalize_time_with_llm(raw_time: str) -> NormalizedTime:
    """Robust normalizer: extracts time from sentences and validates via Pydantic."""
    parser = PydanticOutputParser(pydantic_object=NormalizedTime)
    prompt = PromptTemplate(
        input_variables=["raw_time"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template=(
    """
You convert human-written interview times in *sentences* to 24h HH:MM.
- The input may be a full message like "Can you do it at 2pm or 14:00?" — extract the intended time.
- Valid window: 12:00–18:00 inclusive. If outside window or unparseable, mark invalid.
- Return JSON strictly matching this schema:\n{format_instructions}\n

Examples:
- input: "14:00" -> {{"status":"ok","time":"14:00"}}
- input: "Can you do it at 1pm?" -> {{"status":"ok","time":"13:00"}}
- input: "maybe around 15:30 works" -> {{"status":"ok","time":"15:30"}}
- input: "let's do it at 7pm" -> {{"status":"invalid","time":null}}

Input: {raw_time}
    """
),
    )

    # Heuristic pre-extraction improves robustness; use the candidate if found
    candidate = _preextract_time_token(raw_time)
    print("hii",candidate)
    effective_input = candidate if candidate else raw_time

    # IMPORTANT: pipe prompt -> model -> parser
    result = (prompt | _llm | parser).invoke({"raw_time": effective_input})
    return result


def suggest_alternative_with_llm(requested_time: str, available_slots: List[str]) -> SuggestedTime:
    parser = PydanticOutputParser(pydantic_object=SuggestedTime)
    prompt = PromptTemplate(
        input_variables=["requested_time", "slots"],
        template=(
            """
A recruiter requested {requested_time}, but it's unavailable. From this list, pick the single closest time.
Return JSON strictly matching this schema:\n{format_instructions}\n
Available slots: {slots}
            """
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    result = (prompt | _llm | parser).invoke({
        "requested_time": requested_time,
        "slots": ", ".join(available_slots) if available_slots else "(none)",
    })
    return result


# -------------------- Pure helpers --------------------

def compute_available_slots(booked: List[str], start: str = TIME_START, end: str = TIME_END, step: int = STEP_MINUTES) -> List[str]:
    start_dt = datetime.strptime(start, "%H:%M")
    end_dt = datetime.strptime(end, "%H:%M")
    cur = start_dt
    out: List[str] = []
    while cur <= end_dt:
        s = cur.strftime("%H:%M")
        if s not in booked:
            out.append(s)
        cur += timedelta(minutes=step)
    return out


# -------------------- LangGraph nodes --------------------

def node_load_schedule(state: GraphState) -> Dict[str, Any]:
    df = load_interview_schedule()
    csv_str = df.to_csv(index=False)
    return {"schedule_df_csv": csv_str}


def node_normalize_time(state: GraphState) -> Dict[str, Any]:
    norm = normalize_time_with_llm(state.raw_time)
    print(norm)
    if norm.status == "invalid" or not norm.time:
        return {
            "status": "invalid_time",
            "message": "Invalid time. Interviews can only be scheduled between 12:00 and 18:00.",
            "preferred_time": None,
        }
    return {"preferred_time": norm.time}


def node_check_and_schedule(state: GraphState) -> Dict[str, Any]:
    df = load_interview_schedule()
    booked = df["interview_time"].astype(str).tolist()
    if not state.preferred_time:
        return {}

    if state.preferred_time in booked:
        available = compute_available_slots(booked)
        if not available:
            return {"status": "no_slots", "message": "No slots available today.", "available_slots": []}
        sug = suggest_alternative_with_llm(state.preferred_time, available)
        return {
            "status": "suggested",
            "suggested_time": sug.suggestion,
            "available_slots": available,
            "message": f"Slot {state.preferred_time} is taken. Suggested {sug.suggestion}.",
        }

    # free: schedule it
    row = pd.DataFrame([[state.candidate, state.preferred_time]], columns=["candidate_name", "interview_time"])
    df = pd.concat([df, row], ignore_index=True)
    save_interview_schedule(df)
    return {
        "status": "scheduled",
        "message": f"Scheduled {state.candidate} at {state.preferred_time}.",
        "schedule_df_csv": df.to_csv(index=False),
    }


def node_accept_suggestion(state: GraphState) -> Dict[str, Any]:
    """Optional node when UI confirms the suggested slot."""
    if not state.suggested_time:
        return {}
    df = load_interview_schedule()
    row = pd.DataFrame([[state.candidate, state.suggested_time]], columns=["candidate_name", "interview_time"])
    df = pd.concat([df, row], ignore_index=True)
    save_interview_schedule(df)
    return {
        "status": "scheduled",
        "message": f"Scheduled {state.candidate} at {state.suggested_time}.",
        "schedule_df_csv": df.to_csv(index=False),
    }


# -------------------- Graph builder & runners --------------------

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("load_schedule", node_load_schedule)
    g.add_node("normalize_time", node_normalize_time)
    g.add_node("check_and_schedule", node_check_and_schedule)
    g.add_node("accept_suggestion", node_accept_suggestion)

    g.set_entry_point("load_schedule")
    g.add_edge("load_schedule", "normalize_time")
    g.add_edge("normalize_time", "check_and_schedule")
    # accept_suggestion is optional; UI can invoke it explicitly

    return g.compile()


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict()  # pydantic v1
    return dict(obj)


def run_schedule(candidate: str, raw_time: str) -> Dict[str, Any]:
    """Run the basic path (load -> normalize -> check/schedule)."""
    app = build_graph()
    init = GraphState(candidate=candidate, raw_time=raw_time)
    out = app.invoke(init)
    d = _to_dict(out)
    if d.get("schedule_df_csv"):
        try:
            d["schedule_df"] = pd.read_csv(io.StringIO(d["schedule_df_csv"]))
        except Exception:
            pass
    return d


def accept_suggestion(candidate: str, suggested_time: str) -> Dict[str, Any]:
    """Schedule using a previously suggested time."""
    g = StateGraph(GraphState)
    g.add_node("accept_suggestion", node_accept_suggestion)
    g.set_entry_point("accept_suggestion")
    app = g.compile()
    out = app.invoke(GraphState(candidate=candidate, suggested_time=suggested_time))
    return _to_dict(out)


__all__ = [
    "GraphState",
    "run_schedule",
    "accept_suggestion",
    "load_interview_schedule",
    "save_interview_schedule",
]