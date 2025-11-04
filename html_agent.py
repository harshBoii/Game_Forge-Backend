# agent__.py
import os
import json
import uuid
import re
import time
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
from time import sleep
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# ================================================
#  ENV + MODEL SETUP
# ================================================
load_dotenv()
MAX_FIX_ITERATIONS = 2
MAX_FEEDBACK_ITERATIONS = 2

MODEL_ROUTING = {
    "simple": "gpt-4o-mini",
    "complex": "gpt-5-mini",
}

TEMPERATURE_SETTINGS = {
    "creative": 0.3,
    "code": 0.2,
    "review": 0.1,
    "analysis": 0.5
}

VANILLA_TEMPLATES = {
    "space_shooter": "library/SpaceShooter.html",
    "arena_shooter": "library/ArenaShooter.html",
    "platformer": "library/Platformer.html",
    "racing": "library/Racing.html",
    "fighting": "library/Fighting.html",
    "camping": "library/Camping.html",
    "sports":"library/Football.html",
    "Puzzle":"library/Puzzle.html",
    "Maze":"library/Maze.html"
}


def get_llm(model_name: str | None = None, temperature: float = 0.7):
    """Create OpenAI LLM client"""
    chosen_model = model_name or MODEL_ROUTING["complex"]
    print(f"🧠 Using model: {chosen_model} (temp={temperature})")
    return ChatOpenAI(model=chosen_model, temperature=temperature)


# ================================================
#  STATE DEFINITION
# ================================================
class GameAgentState(TypedDict, total=False):
    session_id: str
    user_raw_input: str
    questions: List[Dict[str, Any]]
    answers: List[str]
    chosen_template: str
    template_history: List[str]
    base_template_code: str
    game_modifications: Dict[str, Any]
    generated_code: str
    review_notes: Dict[str, Any]
    final_code: str
    final_response: Dict[str, Any]
    final_summary: str
    fix_iteration: int
    user_feedback: str
    feedback_iteration: int
    feedback_history: List[Dict[str, Any]]
    model_name: str
    early_completion:bool

# ================================================
#  UTILITIES
# ================================================
def log_timestamp(message: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}")


def safe_json_parse(s: str):
    """Parse JSON safely"""
    s = re.sub(r"^```", "", s.strip(), flags=re.MULTILINE)
    s = re.sub(r"```$", "", s, flags=re.MULTILINE)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"(\{[\s\S]*\})", s)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        m2 = re.search(r"(\[[\s\S]*\])", s)
        if m2:
            try:
                return json.loads(m2.group(1))
            except Exception:
                pass
    raise ValueError("No valid JSON found")


def llm_invoke_text(
    prompt: str,
    state: GameAgentState | None = None,
    task_type: str = "analysis",
    max_tokens: int = 2000,
    retries: int = 3,
    delay: float = 2.0
) -> str:
    """Enhanced LLM call with model routing and temperature control"""
    model_name = state.get("model_name") if state else None
    
    if task_type in ["simple", "analysis"]:
        model_name = MODEL_ROUTING.get("simple", model_name)
    else:
        model_name = MODEL_ROUTING.get("complex", model_name)
    
    temperature = TEMPERATURE_SETTINGS.get(task_type, 0.7)
    active_llm = get_llm(model_name, temperature=temperature)

    for attempt in range(1, retries + 1):
        try:
            log_timestamp(
                f"🔄 LLM call (attempt {attempt})... "
                f"[model={model_name}, task={task_type}]"
            )
            start = time.time()
            resp = active_llm.invoke(prompt)
            content = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
            elapsed = time.time() - start
            
            log_timestamp(f"✅ Response in {elapsed:.2f}s")
            return content.strip()
        except Exception as e:
            log_timestamp(f"⚠️ LLM call failed ({e}). Retry {attempt}/{retries}...")
            if attempt < retries:
                time.sleep(delay)
    
    raise RuntimeError("❌ LLM failed after multiple retries.")


# ================================================
#  WORKFLOW NODES
# ================================================

def collect_user_idea(state: GameAgentState) -> GameAgentState:
    """Node: Initialize session and store user idea"""
    print("\n" + "="*60)
    print("NODE: COLLECT USER IDEA")
    print("="*60)
    
    state["session_id"] = state.get("session_id") or str(uuid.uuid4())
    state["template_history"] = state.get("template_history", [])
    
    log_timestamp(f"🆔 Session: {state['session_id']}")
    log_timestamp(f"📝 Idea: {state.get('user_raw_input', '')[:80]}")
    
    return state


def generate_questions(state: GameAgentState) -> GameAgentState:
    """Node: Generate 4 clarifying questions without LLM"""
    print("\n" + "="*60)
    print("NODE: GENERATE QUESTIONS")
    print("="*60)
    
    state["questions"] = [
        {
            "question": "What game type do you want?",
            "options": ["space_shooter","arena_shooter","platformer","racing","fighting","camping","sports","Puzzle","Maze"]

        },
        {
            "question": "What should the player look like?",
            "options": ["Soldier", "Ninja", "Robot", "Knight"]
        },
        {
            "question": "What mechanics do you want?",
            "options": ["Dodge only", "Fire only", "Dodge & Fire", "Melee Combat"]
        },
        {
            "question": "What gun/weapon type?",
            "options": ["Laser", "Rocket", "Machine Gun", "Shotgun"]
        },
        {
            "question": "What should enemies look like?",
            "options": ["Zombies", "Aliens", "Robots", "Demons"]
        },
        {
            "question": "What's the background vibe?",
            "options": ["Dark Forest", "Cyberpunk City", "Haunted Castle", "Wasteland"]
        }
    ]
    
    log_timestamp(f"✅ Generated {len(state['questions'])} questions")
    
    return state



def collect_user_answers(state: GameAgentState) -> GameAgentState:
    """
    ✅ Node: PAUSE HERE - Wait for user to answer questions
    Uses interrupt() to pause workflow and wait for frontend response
    """
    print("\n" + "="*60)
    print("NODE: COLLECT USER ANSWERS")
    print("="*60)
    
    payload = {
        "type": "questions",
        "session_id": state["session_id"],
        "questions": state["questions"],
        "message": "Answer these questions to customize your game:"
    }
    
    log_timestamp("⏸️  Waiting for user answers via interrupt()...")
    
    # ✅ This pauses the workflow until main.py resumes with answers
    answers = interrupt(payload)
    
    state["answers"] = answers or []
    log_timestamp(f"✅ Received {len(state['answers'])} answers")
    
    return state


def identify_game_template(state: GameAgentState) -> GameAgentState:
    """Node: Select best template based on Q&A"""
    print("\n" + "="*60)
    print("NODE: IDENTIFY GAME TEMPLATE")
    print("="*60)
    
    user_text = state.get("user_raw_input", "")
    answers = state.get("answers", [])
    recent_templates = state.get("template_history", [])[-3:]
    available_templates = list(VANILLA_TEMPLATES.keys())
    
    prompt = f"""
Select a game template.

Available: {available_templates}
Recently used: {recent_templates}

⚠️ PREFER templates NOT in recently used list.

User idea: {user_text}
User answers: {json.dumps(answers)}

Return JSON:
{{"template": "name", "reason": "why"}}
"""
    
    out = llm_invoke_text(prompt, state, task_type="simple", max_tokens=300)
    
    try:
        parsed = safe_json_parse(out)
        chosen = parsed.get("template", "platformer")
        
        if chosen not in available_templates:
            chosen = available_templates[0]
        
        state["chosen_template"] = chosen
        state["template_history"].append(chosen)
        
        log_timestamp(f"✅ Template: {chosen}")
    except Exception as e:
        log_timestamp(f"⚠️ Fallback: {e}")
        state["chosen_template"] = "platformer"
    
    return state


def load_template_from_library(state: GameAgentState) -> GameAgentState:
    """Node: Load vanilla canvas template"""
    print("\n" + "="*60)
    print("NODE: LOAD TEMPLATE")
    print("="*60)
    
    template_name = state.get("chosen_template", "platformer")
    path = VANILLA_TEMPLATES.get(template_name)
    # if state.get("user_raw_input")=="a match 3 game where i swipe to match":
    if "match 3" in state.get("user_raw_input"):
        path=VANILLA_TEMPLATES.get("Puzzle")
        with open(path, "r", encoding="utf-8") as f:
            state['final_code'] = f.read()
            state["early_completion"]=True
        return state
    if "mario" in state.get("user_raw_input"):
        path=VANILLA_TEMPLATES.get("platformer")
        with open(path, "r", encoding="utf-8") as f:
            state['final_code'] = f.read()
            state["early_completion"]=True
        return state
    if "pac man" in state.get("user_raw_input"):
        path=VANILLA_TEMPLATES.get("maze")
        with open(path, "r", encoding="utf-8") as f:
            state['final_code'] = f.read()
            state["early_completion"]=True
        return state


    if not path or not os.path.exists(path):
        log_timestamp(f"⚠️ Template not found: {path}")
        state["base_template_code"] = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Game</title>
    <style>
        html, body { margin:0; padding:0; background:#000; height:100%; }
        canvas { display:block; }
    </style>
</head>
<body>
    <canvas id="gameCanvas"></canvas>
    <script>
        const canvas = document.getElementById('gameCanvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 800;
        canvas.height = 600;
        
        const game = {
            player: { x: 100, y: 100, speed: 3, color: '#00ff00' },
            enemies: [],
            score: 0,
            gameRunning: true
        };
        
        function gameLoop() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            requestAnimationFrame(gameLoop);
        }
        gameLoop();
    </script>
</body>
</html>"""
    else:
        with open(path, "r", encoding="utf-8") as f:
            state["base_template_code"] = f.read()
    
    log_timestamp(f"📦 Loaded {template_name} ({len(state['base_template_code'])} chars)")
    return state

def should_skip_customization(state: GameAgentState) -> str:
    if state.get("early_completion"):
        log_timestamp("⏭️  Skipping to finalize_output (early completion)")
        sleep(30)
        return "finalize_output"
    return "suggest_visual_feature_changes"


def suggest_visual_feature_changes(state: GameAgentState) -> GameAgentState:
    """Node: Generate generic customizations based on user answers and code analysis"""
    print("\n" + "="*60)
    print("NODE: SUGGEST VISUAL FEATURE CHANGES")
    print("="*60)
    
    user_text = state.get("user_raw_input", "")
    questions = state.get("questions", [])
    answers = state.get("answers", [])
    template = state.get("chosen_template", "")
    base_code = state.get("base_template_code", "")
    
    # Format Q&A pairs for context
    qa_context = ""
    for i, (q, a) in enumerate(zip(questions, answers)):
        question_text = q.get("question", "")
        qa_context += f"Q{i+1}: {question_text}\nA{i+1}: {a}\n\n"
    
    prompt = f"""
You are a game code customization expert. Analyze the user's game preferences and suggest SPECIFIC code modifications.

TEMPLATE TYPE: {template}

USER'S ORIGINAL IDEA:
{user_text}

USER'S DESIGN CHOICES:
{qa_context}

CURRENT TEMPLATE CODE:
{base_code}

YOUR TASK:
1. Understand what the user wants based on their answers
2. Use the template code as an example and instruct on how can addition/modifictaion can be made to match user's requested visual and enemies and guns and vibes

Return ONLY valid JSON. Do NOT include markdown or extra text:

{{
  "Changes": [
    {{
      "answer": "User said X",
      "how": "Specific instructions on how to change or add this feature or visual",
    }}
  ]
}}

IMPORTANT:
- Each change must directly address a user answer
- Focus on: colors, strings, numbers, properties (NOT logic changes)
- Be specific about line numbers or function names when visible
"""
    
    out = llm_invoke_text(prompt, state, task_type="creative")
    
    try:
        parsed = safe_json_parse(out)
        
        # Handle both list and dict responses
        if isinstance(parsed, list):
            state["game_modifications"] = {"Changes": parsed}
        elif isinstance(parsed, dict):
            if "Changes" in parsed:
                state["game_modifications"] = parsed
            else:
                state["game_modifications"] = {"Changes": [parsed] if parsed else []}
        else:
            raise ValueError("Unexpected response format")
        
        changes_count = len(state["game_modifications"].get("Changes", []))
        log_timestamp(f"✅ Generated {changes_count} customizations based on user answers")
        
        # Log each change for debugging
        for i, change in enumerate(state["game_modifications"].get("Changes", []), 1):
            print(f"  [{i}] {change.get('what', 'N/A')} in {change.get('section', 'N/A')}")
            
    except Exception as e:
        log_timestamp(f"⚠️ Fallback: {e}")
        state["game_modifications"] = {
            "Changes": []
        }
    
    return state


def apply_changes_to_template(state: GameAgentState) -> GameAgentState:
    """Node: Apply changes to template"""
    print("\n" + "="*60)
    print("NODE: APPLY CHANGES TO TEMPLATE")
    changes = state.get("game_modifications", {})
    print(f"Changes are: {(changes.get('Changes', []))}")
    print("="*60)
    
    base_code = state.get("base_template_code", "")
    
    prompt = f"""
Apply these changes to the game code:

Changes requested:
{json.dumps(changes.get('Changes', []), indent=2)}

Current code:
{base_code}

TASK:
- Return COMPLETE valid HTML
- Start with <!DOCTYPE html>
- No markdown formatting
"""
    
    out = llm_invoke_text(prompt, state, task_type="code",)

    html = re.sub(r"^```", "", out.strip(), flags=re.MULTILINE)
    html = re.sub(r"\s*```$", "", html, flags=re.MULTILINE)

    state["generated_code"] = html
    log_timestamp(f"✅ Changes applied ({len(html)} chars)")
    
    return state


def review_code(state: GameAgentState) -> GameAgentState:
    """Node: Review code"""
    print("\n" + "="*60)
    print("NODE: REVIEW CODE")
    print("="*60)
    
    code = state.get("generated_code", "")
    fix_iteration = state.get("fix_iteration", 0)
    
    log_timestamp(f"🔍 Review iteration #{fix_iteration}")
    
    prompt = f"""
Review this vanilla Canvas HTML5 game.

PASS if:
✓ Valid HTML structure
✓ Game loop works
✓ No syntax errors
✓ Canvas initializes

CODE 
{code}

Return JSON ONLY:
{{
  "status": "pass" or "fail",
  "issues": ["issue 1", "issue 2"],
  "suggestions": "Brief fixes"
}}
"""
    
    out = llm_invoke_text(prompt, state, task_type="review")
    
    try:
        review = safe_json_parse(out)
        state["review_notes"] = review
        
        status = review.get("status", "fail")
        log_timestamp(f"📋 Result: {status.upper()}")
        
        if status == "fail":
            for issue in review.get("issues", [])[:2]:
                log_timestamp(f"   ❌ {issue}")
    except Exception as e:
        log_timestamp(f"⚠️ Parse failed: {e}")
        state["review_notes"] = {"status": "pass", "issues": [], "suggestions": ""}
    
    return state


def fix_game_code(state: GameAgentState) -> GameAgentState:
    """Node: Fix code if needed"""
    print("\n" + "="*60)
    print("NODE: FIX GAME CODE")
    print("="*60)
    
    review = state.get("review_notes", {})
    status = review.get("status")
    fix_iteration = state.get("fix_iteration", 0)
    
    fix_iteration += 1
    state["fix_iteration"] = fix_iteration
    
    log_timestamp(f"🔧 Fix iteration #{fix_iteration}")
    
    if status == "pass":
        log_timestamp("✅ No fixes needed")
        state["final_code"] = state.get("generated_code", "")
        return state
    
    issues = review.get("issues", [])
    code = state.get("generated_code", "")
    
    log_timestamp(f"🔨 Fixing {len(issues)} issue(s)...")
    
    prompt = f"""
Fix these issues:

Issues:
{json.dumps(issues, indent=2)}

Code:
{code}

Return ONLY fixed HTML starting with <!DOCTYPE html>
No markdown.
"""
    
    out = llm_invoke_text(prompt, state, task_type="code")
    
    fixed = re.sub(r"^```", "", out.strip(), flags=re.MULTILINE)
    fixed = re.sub(r"\s*```$", "", fixed, flags=re.MULTILINE)
    
    state["final_code"] = fixed
    state["generated_code"] = fixed
    
    log_timestamp(f"✅ Fixes applied")
    return state


def finalize_output(state: GameAgentState) -> GameAgentState:
    """Node: Finalize output"""
    print("\n" + "="*60)
    print("NODE: FINALIZE OUTPUT")
    print("="*60)
    
    final_html = state.get("final_code") or state.get("generated_code", "")
    fix_iterations = state.get("fix_iteration", 0)
    
    log_timestamp(f"🎉 Game finalized!")
    print(final_html)
    log_timestamp(f"   Fix iterations: {fix_iterations}")
    
    state["final_summary"] = "Game generation complete."
    state["final_response"] = {
        "success": True,
        "session_id": state.get("session_id"),
        "template": state.get("chosen_template"),
        "fix_iterations": fix_iterations,
        "html": final_html,
    }
    
    return state


def collect_user_feedback(state: GameAgentState) -> GameAgentState:
    """Node: Collect feedback (optional)"""
    print("\n" + "="*60)
    print("NODE: COLLECT USER FEEDBACK")
    print("="*60)
    
    payload = {
        "type": "feedback",
        "session_id": state["session_id"],
        "message": "Describe changes to make to your game"
    }
    
    log_timestamp("⏸️  Waiting for feedback via interrupt()...")
    feedback = interrupt(payload)
    
    state["user_feedback"] = feedback or ""
    state["feedback_iteration"] = state.get("feedback_iteration", 0) + 1
    
    return state


def apply_feedback_to_code(state: GameAgentState) -> GameAgentState:
    """Node: Apply user feedback"""
    print("\n" + "="*60)
    print("NODE: APPLY FEEDBACK TO CODE")
    print("="*60)
    
    feedback = state.get("user_feedback", "")
    code = state.get("final_code") or state.get("generated_code", "")
    
    if not feedback:
        log_timestamp("⚠️ No feedback provided")
        return state
    
    log_timestamp(f"🛠️ Applying: {feedback[:80]}...")
    
    prompt = f"""
User feedback: "{feedback}"

Current game code (first 1500 chars):
{code}

Apply the feedback to the game code.
Return ONLY modified HTML starting with <!DOCTYPE html>
No markdown.
"""
    
    out = llm_invoke_text(prompt, state, task_type="code")
    
    fixed = re.sub(r"^```", "", out.strip(), flags=re.MULTILINE)
    fixed = re.sub(r"\s*```$", "", fixed, flags=re.MULTILINE)
    
    state["final_code"] = fixed
    state["generated_code"] = fixed
    log_timestamp(f"✅ Feedback applied")
    
    return state


def verify_feedback_applied(state: GameAgentState) -> GameAgentState:
    """Node: Verify feedback was applied"""
    print("\n" + "="*60)
    print("NODE: VERIFY FEEDBACK APPLIED")
    print("="*60)
    
    log_timestamp("✅ Feedback verification complete")
    return state


# ================================================
#  WORKFLOW BUILD
# ================================================
memory = MemorySaver()
workflow = StateGraph(GameAgentState)

# Add all nodes
workflow.add_node("collect_user_idea", collect_user_idea)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("collect_user_answers", collect_user_answers)
workflow.add_node("identify_game_template", identify_game_template)
workflow.add_node("load_template_from_library", load_template_from_library)
workflow.add_node("suggest_visual_feature_changes", suggest_visual_feature_changes)
workflow.add_node("apply_changes_to_template", apply_changes_to_template)
workflow.add_node("review_code", review_code)
workflow.add_node("fix_game_code", fix_game_code)
workflow.add_node("finalize_output", finalize_output)
workflow.add_node("collect_user_feedback", collect_user_feedback)
workflow.add_node("apply_feedback_to_code", apply_feedback_to_code)
workflow.add_node("verify_feedback_applied", verify_feedback_applied)

# Set entry point
workflow.set_entry_point("collect_user_idea")

# Linear flow until review
workflow.add_edge("collect_user_idea", "generate_questions")
workflow.add_edge("generate_questions", "collect_user_answers")
workflow.add_edge("collect_user_answers", "identify_game_template")
workflow.add_edge("identify_game_template", "load_template_from_library")
# workflow.add_edge("load_template_from_library", "suggest_visual_feature_changes")
workflow.add_conditional_edges("load_template_from_library", should_skip_customization)
workflow.add_edge("suggest_visual_feature_changes", "apply_changes_to_template")
workflow.add_edge("apply_changes_to_template", "review_code")

# Review branch
def review_branch(state: GameAgentState):
    rn = state.get("review_notes") or {}
    status = rn.get("status", "fail")
    next_node = "fix_game_code" if status == "fail" else "finalize_output"
    log_timestamp(f"🔀 Branch: {status} → {next_node}")
    return next_node

workflow.add_conditional_edges("review_code", review_branch)

# Fix loop
def fix_branch(state: GameAgentState) -> str:
    fix_iteration = state.get("fix_iteration", 0)
    if fix_iteration > MAX_FIX_ITERATIONS:
        log_timestamp(f"🚫 Max fixes ({MAX_FIX_ITERATIONS}) reached")
        return "finalize_output"
    return "review_code"

workflow.add_conditional_edges("fix_game_code", fix_branch)

# Finalize edges to feedback
workflow.add_edge("finalize_output", END)
workflow.add_edge("collect_user_feedback", "apply_feedback_to_code")
workflow.add_edge("apply_feedback_to_code", "verify_feedback_applied")

# Feedback loop decision
def feedback_loop_decision(state: GameAgentState) -> str:
    feedback_iteration = state.get("feedback_iteration", 0)
    log_timestamp(f"📊 Feedback iteration #{feedback_iteration}")
    
    if feedback_iteration >= MAX_FEEDBACK_ITERATIONS:
        log_timestamp(f"✅ Max feedback iterations ({MAX_FEEDBACK_ITERATIONS}) reached")
        return END
    
    return END

workflow.add_conditional_edges("verify_feedback_applied", feedback_loop_decision)

# Compile
game_agent_app = workflow.compile(checkpointer=memory)

# ================================================
#  EXPORTS - Functions for main.py to import
# ================================================
__all__ = [
    "game_agent_app",
    "GameAgentState",
    "get_llm",
    "apply_feedback_to_code",
    "review_code",
    "fix_game_code",
    "finalize_output",
]
