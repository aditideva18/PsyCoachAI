"""
PsyCoach AI - Flask + Gemini + JSON storage
All backend logic is intentionally kept in this single main.py file.
"""

import json
import logging
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from google import genai
from google.genai import types
from werkzeug.security import check_password_hash, generate_password_hash


# ============================================================
# 1. ENVIRONMENT AND APPLICATION CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "development-secret-key-change-before-deployment",
)
app.config["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
app.config["GEMINI_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
app.config["JSON_DATA_PATH"] = os.getenv("JSON_DATA_PATH", "data/psycoach_data.json")
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("RENDER", "").lower() == "true"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

gemini_client = (
    genai.Client(api_key=app.config["GEMINI_API_KEY"])
    if app.config["GEMINI_API_KEY"]
    else None
)
if gemini_client is None:
    logger.warning("GEMINI_API_KEY is missing. AI endpoints will return an error.")


# ============================================================
# 2. CONSTANTS
# ============================================================

PRACTICE_SCENARIOS = {
    "job_interview": {
        "title": "Job Interview",
        "description": "Practice answering questions from an interviewer.",
        "role": "interviewer",
        "skills": ["confidence", "clarity", "professionalism", "active listening"],
    },
    "teacher_conversation": {
        "title": "Talking to a Teacher",
        "description": "Practice asking a teacher for help or clarification.",
        "role": "teacher",
        "skills": ["assertiveness", "clarity", "respectfulness"],
    },
    "friend_conflict": {
        "title": "Conflict With a Friend",
        "description": "Practice resolving a disagreement with a friend.",
        "role": "friend",
        "skills": ["empathy", "emotional regulation", "active listening"],
    },
    "networking": {
        "title": "Networking Conversation",
        "description": "Practice introducing yourself to someone new.",
        "role": "new professional contact",
        "skills": ["confidence", "question asking", "conversation flow"],
    },
}

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
AI_PERSONALITIES = ["friendly", "neutral", "busy", "impatient", "supportive", "awkward"]
REWRITE_STYLES = ["calm", "assertive", "empathetic", "professional", "direct"]
COMMUNICATION_SKILLS = [
    "confidence",
    "assertiveness",
    "empathy",
    "clarity",
    "active_listening",
    "emotional_regulation",
    "respectfulness",
    "professionalism",
]

CRISIS_PHRASES = [
    "kill myself",
    "hurt myself",
    "end my life",
    "suicide",
    "want to die",
    "hurt someone",
    "kill someone",
    "immediate danger",
]


# ============================================================
# 3. JSON STORAGE
# ============================================================

DATA_LOCK = threading.RLock()


def empty_data_store() -> dict[str, Any]:
    return {
        "users": [],
        "practice_sessions": [],
        "conversation_messages": [],
        "practice_analyses": [],
        "conflict_analyses": [],
    }


def data_file_path() -> Path:
    return Path(app.config["JSON_DATA_PATH"])


def initialize_storage() -> None:
    path = data_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with DATA_LOCK:
        if not path.exists():
            write_data(empty_data_store())
            return
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                raise ValueError("Storage root must be an object.")
            defaults = empty_data_store()
            changed = False
            for key, default_value in defaults.items():
                if key not in current or not isinstance(current[key], list):
                    current[key] = default_value
                    changed = True
            if changed:
                write_data(current)
        except (json.JSONDecodeError, OSError, ValueError):
            backup = path.with_suffix(f".corrupt-{uuid.uuid4().hex[:8]}.json")
            try:
                path.replace(backup)
            except OSError:
                pass
            write_data(empty_data_store())


def read_data() -> dict[str, Any]:
    initialize_if_missing = not data_file_path().exists()
    if initialize_if_missing:
        initialize_storage()
    with DATA_LOCK:
        try:
            return json.loads(data_file_path().read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.exception("Unable to read JSON storage: %s", exc)
            raise RuntimeError("Application storage could not be read.") from exc


def write_data(data: dict[str, Any]) -> None:
    path = data_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    with DATA_LOCK:
        temporary_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)


def next_integer_id(items: list[dict[str, Any]]) -> int:
    return max((int(item.get("id", 0)) for item in items), default=0) + 1


def mutate_data(mutator):
    with DATA_LOCK:
        data = read_data()
        result = mutator(data)
        write_data(data)
        return result


# ============================================================
# 4. USERS AND AUTHENTICATION
# ============================================================

def create_user(username: str, email: str, password: str) -> int | None:
    username = clean_user_text(username, 80)
    email = clean_user_text(email, 200).lower()
    if not username or not email or len(password) < 8:
        return None

    def operation(data):
        if any(user["email"].lower() == email for user in data["users"]):
            return None
        user_id = next_integer_id(data["users"])
        data["users"].append(
            {
                "id": user_id,
                "username": username,
                "email": email,
                "password_hash": generate_password_hash(password),
                "created_at": utc_timestamp(),
            }
        )
        return user_id

    return mutate_data(operation)


def find_user_by_email(email: str) -> dict[str, Any] | None:
    email = clean_user_text(email, 200).lower()
    return next(
        (deepcopy(user) for user in read_data()["users"] if user["email"].lower() == email),
        None,
    )


def find_user_by_id(user_id: int) -> dict[str, Any] | None:
    return next(
        (deepcopy(user) for user in read_data()["users"] if user["id"] == user_id),
        None,
    )


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            if request.path.startswith("/api/"):
                return create_error_response("Authentication is required.", 401)
            return redirect(url_for("login_page"))
        return view_function(*args, **kwargs)

    return wrapped_view


# ============================================================
# 5. INPUT VALIDATION AND GENERAL HELPERS
# ============================================================

def get_json_request_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def validate_required_fields(
    data: dict[str, Any],
    required_fields: list[str],
) -> tuple[bool, list[str]]:
    missing = []
    for field in required_fields:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return not missing, missing


def clean_user_text(text: Any, maximum_length: int = 10000) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[ \t]+", " ", text.replace("\x00", "")).strip()[:maximum_length]


def validate_score(score: Any) -> int:
    try:
        return max(1, min(10, int(round(float(score)))))
    except (TypeError, ValueError):
        return 1


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_success_response(
    data: Any = None,
    message: str = "Request completed successfully.",
    status_code: int = 200,
):
    return jsonify({"success": True, "message": message, "data": data}), status_code


def create_error_response(
    message: str,
    status_code: int = 400,
    details: Any = None,
):
    return jsonify(
        {"success": False, "message": message, "details": details}
    ), status_code


def is_gemini_available() -> bool:
    return gemini_client is not None


def get_owned_practice_session(
    practice_session_id: int,
    user_id: int,
) -> dict[str, Any] | None:
    return next(
        (
            deepcopy(item)
            for item in read_data()["practice_sessions"]
            if item["id"] == practice_session_id and item["user_id"] == user_id
        ),
        None,
    )


# ============================================================
# 6. SAFETY
# ============================================================

def contains_crisis_language(text: str) -> bool:
    lowered = clean_user_text(text, 20000).lower()
    return any(phrase in lowered for phrase in CRISIS_PHRASES)


def build_crisis_response() -> dict[str, Any]:
    return {
        "crisis_detected": True,
        "message": (
            "This app is not equipped to handle an immediate safety crisis. "
            "Please contact local emergency services now if anyone may be in danger, "
            "and reach out to a trusted adult, guardian, counselor, or qualified "
            "mental-health professional. Do not rely on this AI for emergency help."
        ),
    }


def add_safety_instructions(prompt: str) -> str:
    safety = """
SAFETY RULES:
- You are a communication-practice assistant, not a therapist.
- Never diagnose a mental-health condition or provide medical treatment advice.
- Never claim certainty about another person's private thoughts or motives.
- Do not decide who is morally right or wrong.
- Do not encourage manipulation, retaliation, coercion, stalking, or deception.
- Use tentative language such as "may," "could," and "one possible interpretation."
- If the content suggests immediate danger, self-harm, or harm to others, stop ordinary
  coaching and encourage urgent help from local emergency services and a trusted person.
"""
    return f"{safety}\n\n{prompt}".strip()


# ============================================================
# 7. PROMPT BUILDERS
# ============================================================

def build_roleplay_system_prompt(
    scenario: str,
    difficulty: str,
    personality: str,
) -> str:
    config = PRACTICE_SCENARIOS[scenario]
    return add_safety_instructions(
        f"""
You are roleplaying as a {config["role"]} in the scenario "{config["title"]}".
Personality: {personality}
Difficulty: {difficulty}

Remain in character. Keep each reply to 1-3 short paragraphs. Respond naturally to the
user rather than teaching during the roleplay. Do not score or coach the user until the
practice session ends. For easy difficulty, be cooperative. For medium difficulty, ask
follow-up questions. For hard difficulty, introduce realistic resistance without being
abusive or unsafe.
"""
    )


def conversation_as_text(history: list[dict[str, str]]) -> str:
    return "\n".join(
        f'{item.get("speaker", "unknown").upper()}: {item.get("message", "")}'
        for item in history
    )


def build_roleplay_message_prompt(
    scenario: str,
    difficulty: str,
    personality: str,
    conversation_history: list[dict[str, str]],
    user_message: str,
) -> str:
    return (
        build_roleplay_system_prompt(scenario, difficulty, personality)
        + "\n\nCONVERSATION SO FAR:\n"
        + (conversation_as_text(conversation_history) or "(No previous messages.)")
        + f"\n\nUSER'S NEW MESSAGE:\n{user_message}\n\nReply only as the roleplay character."
    )


def build_practice_analysis_prompt(
    scenario: str,
    conversation_history: list[dict[str, str]],
) -> str:
    scenario_title = PRACTICE_SCENARIOS[scenario]["title"]
    return add_safety_instructions(
        f"""
Analyze the user's communication in this completed roleplay: {scenario_title}.

TRANSCRIPT:
{conversation_as_text(conversation_history)}

Return JSON with exactly these keys:
{{
  "summary": "brief overall assessment",
  "scores": {{
    "confidence": 1,
    "assertiveness": 1,
    "empathy": 1,
    "clarity": 1,
    "active_listening": 1,
    "emotional_regulation": 1,
    "respectfulness": 1,
    "professionalism": 1
  }},
  "strengths": ["specific strength"],
  "improvement_areas": ["specific improvement area"],
  "better_responses": [
    {{"original": "quoted or paraphrased user message", "improved": "better version", "reason": "why"}}
  ],
  "next_practice_goal": "one measurable goal"
}}

Use scores from 1 to 10. Base feedback only on visible language. Be supportive,
specific, and concise.
"""
    )


def build_conflict_analysis_prompt(
    transcript: str,
    user_role: str | None = None,
) -> str:
    return add_safety_instructions(
        f"""
Analyze this conversation transcript as a neutral communication coach.
The user identifies their role as: {user_role or "not specified"}.

TRANSCRIPT:
{transcript}

Use these frameworks carefully:
- Nonviolent Communication: observation, feeling, need, request
- Active listening
- Emotional intelligence
- Cognitive distortions such as all-or-nothing thinking or mind reading
- Criticism, contempt, defensiveness, and stonewalling

Return JSON with exactly these keys:
{{
  "summary": "neutral summary",
  "primary_issue": "main communication problem",
  "emotional_tone": ["tone label"],
  "patterns": [
    {{"type": "pattern", "evidence": "short excerpt or paraphrase", "explanation": "why it matters"}}
  ],
  "healthy_moments": ["healthy moment"],
  "escalation_points": [
    {{"message": "message or paraphrase", "reason": "why it escalated", "alternative": "healthier wording"}}
  ],
  "scores": {{
    "empathy": 1,
    "assertiveness": 1,
    "active_listening": 1,
    "emotional_regulation": 1,
    "respectfulness": 1
  }},
  "possible_needs": ["tentative possible need"],
  "recommended_next_steps": ["specific next step"],
  "disclaimer": "This is communication coaching, not diagnosis or therapy."
}}

Do not infer hidden motives as facts. Do not determine who is right.
"""
    )


def build_rewrite_prompt(
    original_message: str,
    rewrite_style: str,
    context: str | None = None,
) -> str:
    return add_safety_instructions(
        f"""
Rewrite the message using a {rewrite_style} communication style.

CONTEXT:
{context or "No additional context provided."}

ORIGINAL MESSAGE:
{original_message}

Return JSON:
{{
  "rewritten_message": "rewritten version",
  "changes": ["specific change"],
  "why_it_may_help": "brief explanation"
}}

Preserve the user's legitimate boundary or request. Do not make the message manipulative.
"""
    )


def build_perspective_prompt(transcript: str, selected_message: str) -> str:
    return add_safety_instructions(
        f"""
Explain several reasonable ways another person might interpret the selected message.
Do not claim to know what they truly thought.

FULL TRANSCRIPT:
{transcript}

SELECTED MESSAGE:
{selected_message}

Return JSON:
{{
  "possible_interpretations": [
    {{"interpretation": "possible interpretation", "why": "language cue"}}
  ],
  "ambiguities": ["what is unclear"],
  "clearer_version": "a clearer, respectful rewrite"
}}
"""
    )


# ============================================================
# 8. GEMINI SERVICE
# ============================================================

def call_gemini_text(prompt: str) -> str:
    if gemini_client is None:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY.")
    try:
        response = gemini_client.models.generate_content(
            model=app.config["GEMINI_MODEL"],
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=800,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        return text
    except Exception as exc:
        logger.exception("Gemini text request failed: %s", exc)
        raise RuntimeError("Gemini could not generate a response.") from exc


def strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def call_gemini_json(
    prompt: str,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if gemini_client is None:
        raise RuntimeError("Gemini is not configured. Add GEMINI_API_KEY.")
    try:
        config_kwargs: dict[str, Any] = {
            "temperature": 0.3,
            "max_output_tokens": 1800,
            "response_mime_type": "application/json",
        }
        if response_schema:
            config_kwargs["response_schema"] = response_schema
        response = gemini_client.models.generate_content(
            model=app.config["GEMINI_MODEL"],
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw = strip_json_fences(response.text or "")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini JSON response must be an object.")
        return parsed
    except Exception as exc:
        logger.exception("Gemini JSON request failed: %s", exc)
        raise RuntimeError("Gemini could not generate a structured response.") from exc


def generate_roleplay_reply(
    scenario: str,
    difficulty: str,
    personality: str,
    conversation_history: list[dict[str, str]],
    user_message: str,
) -> dict[str, Any]:
    prompt = build_roleplay_message_prompt(
        scenario,
        difficulty,
        personality,
        conversation_history,
        user_message,
    )
    return {"reply": call_gemini_text(prompt), "generated_at": utc_timestamp()}


def normalize_scores(analysis: dict[str, Any], skills: list[str]) -> dict[str, Any]:
    raw_scores = analysis.get("scores")
    if not isinstance(raw_scores, dict):
        raw_scores = {}
    analysis["scores"] = {
        skill: validate_score(raw_scores.get(skill, 1))
        for skill in skills
    }
    return analysis


def analyze_practice_session(
    scenario: str,
    conversation_history: list[dict[str, str]],
) -> dict[str, Any]:
    analysis = call_gemini_json(
        build_practice_analysis_prompt(scenario, conversation_history)
    )
    return normalize_scores(analysis, COMMUNICATION_SKILLS)


def analyze_conflict(
    transcript: str,
    user_role: str | None = None,
) -> dict[str, Any]:
    analysis = call_gemini_json(build_conflict_analysis_prompt(transcript, user_role))
    return normalize_scores(
        analysis,
        ["empathy", "assertiveness", "active_listening", "emotional_regulation", "respectfulness"],
    )


def rewrite_communication(
    original_message: str,
    rewrite_style: str,
    context: str | None = None,
) -> dict[str, Any]:
    return call_gemini_json(
        build_rewrite_prompt(original_message, rewrite_style, context)
    )


def explain_other_perspective(
    transcript: str,
    selected_message: str,
) -> dict[str, Any]:
    return call_gemini_json(build_perspective_prompt(transcript, selected_message))


# ============================================================
# 9. JSON SAVE AND READ FUNCTIONS
# ============================================================

def create_practice_session(
    user_id: int,
    scenario: str,
    difficulty: str,
    personality: str,
) -> int | None:
    def operation(data):
        session_id = next_integer_id(data["practice_sessions"])
        data["practice_sessions"].append(
            {
                "id": session_id,
                "user_id": user_id,
                "scenario": scenario,
                "difficulty": difficulty,
                "personality": personality,
                "status": "active",
                "created_at": utc_timestamp(),
                "completed_at": None,
            }
        )
        return session_id

    return mutate_data(operation)


def save_conversation_message(
    practice_session_id: int,
    speaker: str,
    message: str,
) -> int | None:
    speaker = speaker if speaker in {"user", "ai"} else "unknown"

    def operation(data):
        message_id = next_integer_id(data["conversation_messages"])
        data["conversation_messages"].append(
            {
                "id": message_id,
                "practice_session_id": practice_session_id,
                "speaker": speaker,
                "message": clean_user_text(message, 15000),
                "created_at": utc_timestamp(),
            }
        )
        return message_id

    return mutate_data(operation)


def get_conversation_history(
    practice_session_id: int,
) -> list[dict[str, str]]:
    messages = [
        item
        for item in read_data()["conversation_messages"]
        if item["practice_session_id"] == practice_session_id
    ]
    messages.sort(key=lambda item: item["id"])
    return [
        {
            "speaker": item["speaker"],
            "message": item["message"],
            "created_at": item["created_at"],
        }
        for item in messages
    ]


def save_practice_analysis(
    practice_session_id: int,
    analysis: dict[str, Any],
) -> int | None:
    def operation(data):
        analysis_id = next_integer_id(data["practice_analyses"])
        data["practice_analyses"].append(
            {
                "id": analysis_id,
                "practice_session_id": practice_session_id,
                "analysis": analysis,
                "created_at": utc_timestamp(),
            }
        )
        for item in data["practice_sessions"]:
            if item["id"] == practice_session_id:
                item["status"] = "completed"
                item["completed_at"] = utc_timestamp()
                break
        return analysis_id

    return mutate_data(operation)


def save_conflict_analysis(
    user_id: int,
    transcript: str,
    analysis: dict[str, Any],
) -> int | None:
    def operation(data):
        analysis_id = next_integer_id(data["conflict_analyses"])
        data["conflict_analyses"].append(
            {
                "id": analysis_id,
                "user_id": user_id,
                "transcript": transcript,
                "analysis": analysis,
                "created_at": utc_timestamp(),
            }
        )
        return analysis_id

    return mutate_data(operation)


def get_analysis_report(
    analysis_id: int,
    user_id: int,
) -> dict[str, Any] | None:
    data = read_data()
    session_map = {item["id"]: item for item in data["practice_sessions"]}
    for item in data["practice_analyses"]:
        session_record = session_map.get(item["practice_session_id"])
        if item["id"] == analysis_id and session_record and session_record["user_id"] == user_id:
            return {"type": "practice", **deepcopy(item), "session": deepcopy(session_record)}
    for item in data["conflict_analyses"]:
        if item["id"] == analysis_id and item["user_id"] == user_id:
            return {"type": "conflict", **deepcopy(item)}
    return None


def get_user_dashboard_data(user_id: int) -> dict[str, Any]:
    data = read_data()
    sessions = [s for s in data["practice_sessions"] if s["user_id"] == user_id]
    session_ids = {s["id"] for s in sessions}
    analyses = [
        a for a in data["practice_analyses"]
        if a["practice_session_id"] in session_ids
    ]
    totals = {skill: 0 for skill in COMMUNICATION_SKILLS}
    counts = {skill: 0 for skill in COMMUNICATION_SKILLS}
    for item in analyses:
        scores = item.get("analysis", {}).get("scores", {})
        for skill in COMMUNICATION_SKILLS:
            if skill in scores:
                totals[skill] += validate_score(scores[skill])
                counts[skill] += 1
    averages = {
        skill: round(totals[skill] / counts[skill], 1) if counts[skill] else 0
        for skill in COMMUNICATION_SKILLS
    }
    recent = sorted(sessions, key=lambda item: item["created_at"], reverse=True)[:5]
    return {
        "sessions_completed": sum(s["status"] == "completed" for s in sessions),
        "active_sessions": sum(s["status"] == "active" for s in sessions),
        "conflict_analyses": sum(
            item["user_id"] == user_id for item in data["conflict_analyses"]
        ),
        "average_scores": averages,
        "recent_sessions": recent,
    }


def get_user_history(user_id: int) -> list[dict[str, Any]]:
    data = read_data()
    history: list[dict[str, Any]] = []
    sessions = {s["id"]: s for s in data["practice_sessions"] if s["user_id"] == user_id}
    for analysis in data["practice_analyses"]:
        practice_session = sessions.get(analysis["practice_session_id"])
        if practice_session:
            history.append(
                {
                    "type": "practice",
                    "id": analysis["id"],
                    "created_at": analysis["created_at"],
                    "title": PRACTICE_SCENARIOS.get(
                        practice_session["scenario"], {}
                    ).get("title", practice_session["scenario"]),
                    "analysis": analysis["analysis"],
                }
            )
    for analysis in data["conflict_analyses"]:
        if analysis["user_id"] == user_id:
            history.append(
                {
                    "type": "conflict",
                    "id": analysis["id"],
                    "created_at": analysis["created_at"],
                    "title": "Conflict Analysis",
                    "analysis": analysis["analysis"],
                }
            )
    return sorted(history, key=lambda item: item["created_at"], reverse=True)


# ============================================================
# 10. PAGE ROUTES
# ============================================================

@app.context_processor
def inject_current_user():
    user = find_user_by_id(session["user_id"]) if "user_id" in session else None
    return {"current_user": user}


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        return render_template("register.html")

    data = get_json_request_body() if request.is_json else request.form.to_dict()
    valid, missing = validate_required_fields(data, ["username", "email", "password"])
    if not valid:
        if request.is_json:
            return create_error_response("Required fields are missing.", 400, missing)
        return render_template("register.html", error="Please complete every field."), 400

    password = str(data["password"])
    if len(password) < 8:
        message = "Password must contain at least 8 characters."
        return (
            create_error_response(message, 400)
            if request.is_json
            else (render_template("register.html", error=message), 400)
        )

    user_id = create_user(data["username"], data["email"], password)
    if user_id is None:
        message = "That email is already registered or the input is invalid."
        return (
            create_error_response(message, 409)
            if request.is_json
            else (render_template("register.html", error=message), 409)
        )

    session.clear()
    session["user_id"] = user_id
    if request.is_json:
        return create_success_response({"user_id": user_id}, "Registration successful.", 201)
    return redirect(url_for("dashboard_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "GET":
        return render_template("login.html")

    data = get_json_request_body() if request.is_json else request.form.to_dict()
    valid, missing = validate_required_fields(data, ["email", "password"])
    if not valid:
        if request.is_json:
            return create_error_response("Email and password are required.", 400, missing)
        return render_template("login.html", error="Enter your email and password."), 400

    user = find_user_by_email(data["email"])
    if not user or not check_password_hash(user["password_hash"], str(data["password"])):
        message = "Invalid email or password."
        return (
            create_error_response(message, 401)
            if request.is_json
            else (render_template("login.html", error=message), 401)
        )

    session.clear()
    session["user_id"] = user["id"]
    if request.is_json:
        return create_success_response({"user_id": user["id"]}, "Login successful.")
    return redirect(url_for("dashboard_page"))


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("home_page"))


@app.route("/practice")
@login_required
def practice_page():
    return render_template(
        "practice.html",
        scenarios=PRACTICE_SCENARIOS,
        difficulties=DIFFICULTY_LEVELS,
        personalities=AI_PERSONALITIES,
    )


@app.route("/conflict")
@login_required
def conflict_page():
    return render_template("conflict.html", rewrite_styles=REWRITE_STYLES)


@app.route("/dashboard")
@login_required
def dashboard_page():
    return render_template(
        "dashboard.html",
        dashboard_data=get_user_dashboard_data(session["user_id"]),
    )


@app.route("/history")
@login_required
def history_page():
    return render_template(
        "history.html",
        history=get_user_history(session["user_id"]),
    )


@app.route("/report/<int:analysis_id>")
@login_required
def report_page(analysis_id: int):
    report = get_analysis_report(analysis_id, session["user_id"])
    if report is None:
        return render_template("404.html"), 404
    return render_template("report.html", report=report)


# ============================================================
# 11. HEALTH AND PUBLIC API
# ============================================================

@app.route("/health")
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "service": "PsyCoach AI",
            "gemini_configured": is_gemini_available(),
            "storage_path": str(data_file_path()),
            "timestamp": utc_timestamp(),
        }
    )


@app.route("/api/scenarios")
def api_get_scenarios():
    return create_success_response(PRACTICE_SCENARIOS, "Scenarios retrieved.")


# ============================================================
# 12. PRACTICE API
# ============================================================

@app.route("/api/practice/start", methods=["POST"])
@login_required
def api_start_practice():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["scenario", "difficulty", "personality"])
    if not valid:
        return create_error_response("Required fields are missing.", 400, missing)

    scenario = data["scenario"]
    difficulty = data["difficulty"]
    personality = data["personality"]
    if scenario not in PRACTICE_SCENARIOS:
        return create_error_response("Invalid scenario.", 400)
    if difficulty not in DIFFICULTY_LEVELS:
        return create_error_response("Invalid difficulty.", 400)
    if personality not in AI_PERSONALITIES:
        return create_error_response("Invalid personality.", 400)

    practice_session_id = create_practice_session(
        session["user_id"], scenario, difficulty, personality
    )
    try:
        opening = call_gemini_text(
            build_roleplay_system_prompt(scenario, difficulty, personality)
            + "\n\nBegin the roleplay with a natural opening line. Reply only in character."
        )
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)

    save_conversation_message(practice_session_id, "ai", opening)
    return create_success_response(
        {"practice_session_id": practice_session_id, "reply": opening},
        "Practice session started.",
        201,
    )


@app.route("/api/practice/message", methods=["POST"])
@login_required
def api_send_practice_message():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["practice_session_id", "message"])
    if not valid:
        return create_error_response("Required fields are missing.", 400, missing)

    try:
        practice_session_id = int(data["practice_session_id"])
    except (TypeError, ValueError):
        return create_error_response("Invalid practice session ID.", 400)

    practice_record = get_owned_practice_session(practice_session_id, session["user_id"])
    if not practice_record:
        return create_error_response("Practice session was not found.", 404)
    if practice_record["status"] != "active":
        return create_error_response("This practice session has already ended.", 409)

    message = clean_user_text(data["message"], 4000)
    if not message:
        return create_error_response("Message cannot be empty.", 400)
    if contains_crisis_language(message):
        return create_success_response(build_crisis_response(), "Safety response returned.")

    history_before = get_conversation_history(practice_session_id)
    save_conversation_message(practice_session_id, "user", message)
    try:
        result = generate_roleplay_reply(
            practice_record["scenario"],
            practice_record["difficulty"],
            practice_record["personality"],
            history_before,
            message,
        )
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)

    save_conversation_message(practice_session_id, "ai", result["reply"])
    return create_success_response(result, "Roleplay reply generated.")


@app.route("/api/practice/end", methods=["POST"])
@login_required
def api_end_practice():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["practice_session_id"])
    if not valid:
        return create_error_response("Practice session ID is required.", 400, missing)

    try:
        practice_session_id = int(data["practice_session_id"])
    except (TypeError, ValueError):
        return create_error_response("Invalid practice session ID.", 400)

    practice_record = get_owned_practice_session(practice_session_id, session["user_id"])
    if not practice_record:
        return create_error_response("Practice session was not found.", 404)

    history = get_conversation_history(practice_session_id)
    user_messages = [m for m in history if m["speaker"] == "user"]
    if not user_messages:
        return create_error_response("Send at least one message before ending.", 400)

    try:
        analysis = analyze_practice_session(practice_record["scenario"], history)
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)

    analysis_id = save_practice_analysis(practice_session_id, analysis)
    return create_success_response(
        {"analysis_id": analysis_id, "analysis": analysis},
        "Practice session analyzed.",
    )


# ============================================================
# 13. CONFLICT API
# ============================================================

@app.route("/api/conflict/analyze", methods=["POST"])
@login_required
def api_analyze_conflict():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["transcript"])
    if not valid:
        return create_error_response("A transcript is required.", 400, missing)

    transcript = clean_user_text(data["transcript"], 15000)
    user_role = clean_user_text(data.get("user_role", ""), 100) or None
    if len(transcript) < 20:
        return create_error_response("Please provide a longer conversation transcript.", 400)
    if contains_crisis_language(transcript):
        return create_success_response(build_crisis_response(), "Safety response returned.")

    try:
        analysis = analyze_conflict(transcript, user_role)
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)

    analysis_id = save_conflict_analysis(session["user_id"], transcript, analysis)
    return create_success_response(
        {"analysis_id": analysis_id, "analysis": analysis},
        "Conflict analyzed.",
        201,
    )


@app.route("/api/conflict/rewrite", methods=["POST"])
@login_required
def api_rewrite_message():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["message", "style"])
    if not valid:
        return create_error_response("Message and style are required.", 400, missing)

    message = clean_user_text(data["message"], 4000)
    style = clean_user_text(data["style"], 50).lower()
    context = clean_user_text(data.get("context", ""), 6000) or None
    if style not in REWRITE_STYLES:
        return create_error_response(
            f"Style must be one of: {', '.join(REWRITE_STYLES)}.", 400
        )
    if contains_crisis_language(message):
        return create_success_response(build_crisis_response(), "Safety response returned.")

    try:
        result = rewrite_communication(message, style, context)
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)
    return create_success_response(result, "Message rewritten.")


@app.route("/api/conflict/perspective", methods=["POST"])
@login_required
def api_perspective_switch():
    data = get_json_request_body()
    valid, missing = validate_required_fields(data, ["transcript", "selected_message"])
    if not valid:
        return create_error_response("Transcript and selected message are required.", 400, missing)

    transcript = clean_user_text(data["transcript"], 15000)
    selected_message = clean_user_text(data["selected_message"], 4000)
    if contains_crisis_language(transcript):
        return create_success_response(build_crisis_response(), "Safety response returned.")

    try:
        result = explain_other_perspective(transcript, selected_message)
    except RuntimeError as exc:
        return create_error_response(str(exc), 503)
    return create_success_response(result, "Perspective analysis generated.")


# ============================================================
# 14. DASHBOARD API
# ============================================================

@app.route("/api/dashboard")
@login_required
def api_dashboard_data():
    return create_success_response(
        get_user_dashboard_data(session["user_id"]),
        "Dashboard data retrieved.",
    )


@app.route("/api/history")
@login_required
def api_history():
    return create_success_response(
        get_user_history(session["user_id"]),
        "History retrieved.",
    )


# ============================================================
# 15. ERROR HANDLERS
# ============================================================

@app.errorhandler(400)
def handle_bad_request(error):
    if request.path.startswith("/api/"):
        return create_error_response("The request was invalid.", 400)
    return render_template("404.html"), 400


@app.errorhandler(404)
def handle_not_found(error):
    if request.path.startswith("/api/"):
        return create_error_response("The requested endpoint was not found.", 404)
    return render_template("404.html"), 404


@app.errorhandler(413)
def handle_file_too_large(error):
    return create_error_response("The submitted content is too large.", 413)


@app.errorhandler(500)
def handle_server_error(error):
    logger.exception("Unexpected server error: %s", error)
    if request.path.startswith("/api/"):
        return create_error_response("An unexpected server error occurred.", 500)
    return render_template("500.html"), 500


# ============================================================
# 16. STARTUP
# ============================================================

initialize_storage()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
