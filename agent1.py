# agent.py
import os
import io
import glob
import json
from typing import List, Dict, Tuple
import pandas as pd

from dotenv import load_dotenv
from pydantic import BaseModel, Field, conint, confloat
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, END

load_dotenv()

# =====================
# STATE
# =====================

class GraphState(dict):
    jd_text: str                      # JD text
    resumes: Dict[str, str]           # {filename: plain_text}
    rubric: dict                      # generated scoring rubric
    scores: List[dict]                # list of ResumeScore dicts
    top_n: int                        # number of candidates to return
    eval_feedback: Dict[str, str]     # per-file feedback to improve next pass
    loop_i: int                       # evaluation loop iteration (0..3)

# =====================
# Pydantic Schemas
# =====================

class Criterion(BaseModel):
    name: str
    description: str
    weight: conint(ge=5, le=50)  # percent weight per criterion (5–50)

class Rubric(BaseModel):
    role_title: str
    criteria: List[Criterion]
    note_to_evaluator: str = Field(default="Score each criterion 0–10; justify briefly.")

class ResumeScore(BaseModel):
    candidate_file: str
    criterion_scores: Dict[str, confloat(ge=0, le=10)]
    total_score: confloat(ge=0, le=100)
    recommend_interview: bool
    strengths: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    rationale: str

# =====================
# Helpers
# =====================

def _read_pdf_bytes(b: bytes) -> str:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(b)
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
        return "\n".join(p.page_content for p in pages)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _read_uploaded_file(name: str, data: bytes) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext == ".pdf":
        return _read_pdf_bytes(data)
    elif ext in (".txt", ".md"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    else:
        raise ValueError(f"Unsupported upload format for {name}. Use PDF or TXT.")

# =====================
# LangGraph Nodes
# =====================

def build_rubric_node(state: GraphState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=Rubric)
    format_instructions = parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["jd", "format_instructions"],
        template=(
            "You are a recruiting evaluator. From the JD below, create a concise scoring rubric.\n"
            "Rules:\n"
            "- 4 to 6 criteria only, each with a distinct focus (skills, experience, domain, education/certs, impact, communication/collaboration).\n"
            "- Each criterion has a weight between 5 and 50. Sum of weights must be 100.\n"
            "- Criteria must be general enough to apply to most resumes for this role.\n\n"
            "JOB DESCRIPTION:\n{jd}\n\n"
            "Respond as JSON matching this schema exactly:\n{format_instructions}"
        ),
    )
    rubric: Rubric = (prompt | llm | parser).invoke(
        {"jd": state["jd_text"], "format_instructions": format_instructions}
    )

    # Safety: enforce exact sum 100 by re-normalizing tiny drift
    total_w = sum(c.weight for c in rubric.criteria)
    if total_w != 100:
        # normalize proportionally and round to nearest int with final fixup
        raw = [c.weight for c in rubric.criteria]
        norm = [max(5, min(50, round(w * 100.0 / total_w))) for w in raw]
        delta = 100 - sum(norm)
        # fix rounding drift on the largest-weight criteria
        if delta != 0:
            idx = max(range(len(norm)), key=lambda i: norm[i])
            norm[idx] = max(5, min(50, norm[idx] + delta))
        for c, w in zip(rubric.criteria, norm):
            c.weight = int(w)
    #print(rubric.dict())

    return {"rubric": rubric.dict()}

def score_all_resumes_node(state: GraphState):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResumeScore)
    format_instructions = parser.get_format_instructions()

    rubric = Rubric(**state["rubric"])
    feedback_map: Dict[str, str] = state.get("eval_feedback", {}) or {}
    #print(feedback_map)
    scores: List[dict] = []

    base_prompt = PromptTemplate(
        input_variables=["resume_text", "rubric_json", "file_name", "format_instructions", "feedback"],
        template=(
            "You are a fair, strict resume evaluator.\n"
            "Rubric (JSON):\n{rubric_json}\n\n"
            "Resume (plain text):\n{resume_text}\n\n"
            "Reviewer feedback from a previous pass (may be empty):\n{feedback}\n\n"
            "Instructions:\n"
            "- For each rubric criterion, assign a score 0–10.\n"
            "- Compute total_score = sum(criterion_score * (weight/10)) where weight is % from rubric.\n"
            "  Example: 8/10 with 25% weight -> 8*2.5 = 20 toward total.\n"
            "- total_score must be 0–100 (float). Recheck your math before responding.\n"
            "- recommend_interview = True only if evidence supports it.\n"
            "- Provide 2–4 strengths and 1–3 concerns.\n"
            "- Be specific and reference evidence from the resume.\n\n"
            "Return JSON that matches this schema:\n{format_instructions}\n\n"
            "candidate_file={file_name}"
        ),
    )

    for fname, text in state["resumes"].items():
        payload = {
            "resume_text": text,
            "rubric_json": json.dumps(rubric.dict(), ensure_ascii=False),
            "file_name": fname,
            "format_instructions": format_instructions,
            "feedback": feedback_map.get(fname, "").strip(),
        }
        try:
            print(payload)
            result: ResumeScore = (base_prompt | llm | parser).invoke(payload)
            # Clamp + trust-but-verify the math
            # If LLM math is off, we'll correct and feed back in the evaluator
            total = float(result.total_score)
            result.total_score = max(0.0, min(100.0, total))
            scores.append(result.dict())
        except Exception as e:
            scores.append({
                "candidate_file": fname,
                "criterion_scores": {},
                "total_score": 0.0,
                "recommend_interview": False,
                "strengths": [],
                "concerns": [f"Scoring failed: {e}"],
                "rationale": "Parsing or validation failed."
            })

    # reset feedback; next node will set it
    return {"scores": scores, "eval_feedback": {}}

def evaluate_scores_node(state: GraphState):
    """
    1) Verifies weight sum == 100.
    2) Recomputes each candidate's total from criterion scores & weights.
    3) Uses LLM as an 'auditor' to give targeted feedback (math & rationale).
    Always runs and increments loop_i; forces 3 passes before proceeding.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rubric = Rubric(**state["rubric"])
    weights = {c.name: c.weight for c in rubric.criteria}
    weight_sum = sum(weights.values())

    feedback: Dict[str, str] = {}
    corrected_scores: List[dict] = []

    # Prepare an auditor prompt once
    auditor_prompt = PromptTemplate(
        input_variables=["rubric_json", "score_json"],
        template=(
            "You are an auditor verifying resume scoring quality.\n"
            "Rubric:\n{rubric_json}\n\n"
            "Current Score JSON for one candidate:\n{score_json}\n\n"
            "Tasks:\n"
            "1) Check math: total_score must equal sum(criterion_scores[name]*(weight/10)).\n"
            "2) Check that each criterion score 0–10 has evidence per rationale/strengths/concerns.\n"
            "3) If anything seems off, briefly say what to fix and propose corrected numbers.\n"
            "Reply with a short, actionable bullet list (<120 words)."
        )
    )

    # Recompute totals & collect math feedback
    for s in state["scores"]:
        fname = s.get("candidate_file", "unknown")
        criterion_scores: Dict[str, float] = s.get("criterion_scores", {}) or {}
        # recompute
        recomputed = 0.0
        missing = []
        for cname, w in weights.items():
            if cname in criterion_scores:
                recomputed += float(criterion_scores[cname]) * (w / 10.0)
            else:
                missing.append(cname)

        math_issues = []
        if weight_sum != 100:
            math_issues.append(f"Rubric weights sum to {weight_sum}, must be 100.")
        # numerical difference tolerance
        if abs((s.get("total_score", 0.0) or 0.0) - recomputed) > 0.25:
            math_issues.append(
                f"total_score mismatch: reported={s.get('total_score'):.2f}, expected={recomputed:.2f}."
            )
        if missing:
            math_issues.append(f"Missing criterion scores for: {', '.join(missing)}.")

        # LLM auditor feedback (kept brief)
        try:
            audit_text = (auditor_prompt | llm).invoke({
                "rubric_json": json.dumps(rubric.dict(), ensure_ascii=False),
                "score_json": json.dumps(s, ensure_ascii=False)
            }).content.strip()
        except Exception as e:
            audit_text = f"Auditor LLM failed: {e}"

        combined_feedback = ""
        if math_issues:
            combined_feedback += "Math issues:\n- " + "\n- ".join(math_issues) + "\n\n"
        combined_feedback += "Quality review:\n" + audit_text

        feedback[fname] = combined_feedback

        # Store corrected (non-destructive): keep original but snap total to recomputed if off
        s_fixed = dict(s)
        s_fixed["total_score"] = float(max(0.0, min(100.0, recomputed)))
        corrected_scores.append(s_fixed)

    # Increment loop counter, always enforce three passes minimum
    loop_i = int(state.get("loop_i", 0)) + 1

    # We keep the corrected math in scores so ranking isn't skewed if user inspects interim outputs
    return {
        "scores": corrected_scores,
        "eval_feedback": feedback,
        "loop_i": loop_i,
    }

def rank_and_shortlist_node(state: GraphState):
    df = pd.DataFrame(state["scores"]).copy()
    if df.empty:
        return {"top_n": []}

    df["rec_int"] = df["recommend_interview"].astype(int)
    df = df.sort_values(["rec_int", "total_score"], ascending=[False, False])

    k = int(state.get("top_n", 2) or 2)
    top_n = df.head(k).drop(columns=["rec_int"]).to_dict(orient="records")

    return {"top_n": top_n, "scores": df.drop(columns=["rec_int"]).to_dict(orient="records")}

# =====================
# Build Graph
# =====================

def _route_after_eval(state: GraphState):
    # Force at least 3 full audit cycles before proceeding
    if int(state.get("loop_i", 0)) < 2:
        return "score_all"
    return "rank_and_shortlist"

def _build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("build_rubric", build_rubric_node)
    workflow.add_node("score_all", score_all_resumes_node)
    workflow.add_node("evaluate_scores", evaluate_scores_node)
    workflow.add_node("rank_and_shortlist", rank_and_shortlist_node)

    workflow.set_entry_point("build_rubric")
    workflow.add_edge("build_rubric", "score_all")
    workflow.add_edge("score_all", "evaluate_scores")
    workflow.add_conditional_edges("evaluate_scores", _route_after_eval, {
        "score_all": "score_all",
        "rank_and_shortlist": "rank_and_shortlist",
    })
    workflow.add_edge("rank_and_shortlist", END)

    return workflow.compile()

_GRAPH = _build_graph()

# =====================
# Public API
# =====================

def shortlist_with_streamlit_inputs(
    jd_text: str,
    uploaded_files: List[Tuple[str, bytes]],
    top_n: int,
) -> Tuple[List[dict], List[dict]]:
    """
    Args:
        jd_text: the JD plain text
        uploaded_files: list of (filename, bytes) for resumes
        top_n: number of candidates to shortlist

    Returns:
        top_n, scores (both as list[dict])
    """
    if not jd_text or not uploaded_files:
        raise ValueError("jd_text and uploaded_files are required.")

    resumes: Dict[str, str] = {}
    for name, data in uploaded_files:
        resumes[name] = _read_uploaded_file(name, data)

    initial = {
        "jd_text": jd_text,
        "resumes": resumes,
        "top_n": int(top_n),
        "loop_i": 0,
        "eval_feedback": {},
    }
    out = _GRAPH.invoke(initial)

    top = out.get("top_n", [])
    scores = out.get("scores", [])
    return top, scores
