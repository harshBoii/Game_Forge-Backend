import os
import json
import uuid
import re
import time
from typing import TypedDict, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv


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


def get_llm(model_name: str | None = None):
    """Create an Anthropic LLM client with a specified model name."""
    chosen_model = model_name 
    print(f"🧠 Using model: {chosen_model}")
    return ChatOpenAI(model=chosen_model)

llm=""

# ================================================
#  STATE DEFINITION
# ================================================
class GameAgentState(TypedDict, total=False):
    session_id: str
    user_raw_input: str
    questions: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
    chosen_template: str
    base_template_code: str
    game_modifications: Dict[str, Any]
    generated_code: str
    review_notes: Dict[str, Any]
    final_code: str
    final_response: Dict[str, Any]
    fix_iteration: int
    user_feedback: str  # ✅ Added
    feedback_iteration: int  # ✅ Added
    feedback_history: List[Dict[str, Any]]  # ✅ Added
    model_name : str


# ================================================
#  UTILITIES
# ================================================
def log_timestamp(message: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {message}")



def safe_json_parse(s: str):
    """Parse JSON safely even if model output is noisy."""
    s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.MULTILINE)
    s = re.sub(r"```$", "", s.strip(), flags=re.MULTILINE)
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
    raise ValueError("No valid JSON found in string")



def llm_invoke_text(prompt: str, state: GameAgentState | None = None, retries: int = 3, delay: float = 2.0) -> str:
    """Call the LLM dynamically with retry mechanism, using the model from state if available."""
    model_name = None

    # ✅ Try to use model from state if provided
    if state and "model_name" in state:
        model_name = state["model_name"]

    # ✅ Fallback to global default
    active_llm = get_llm(model_name)

    for attempt in range(1, retries + 1):
        try:
            log_timestamp(f"🔄 LLM call (attempt {attempt})... [model={model_name or 'default'}]")
            start = time.time()
            resp = active_llm.invoke(prompt)
            content = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
            elapsed = time.time() - start
            log_timestamp(f"✅ Response in {elapsed:.2f}s")
            return content.strip()
        except Exception as e:
            log_timestamp(f"⚠️ LLM call failed ({e}). Retry in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("❌ LLM failed after multiple retries.")

# ================================================
#  NODES
# ================================================


def collect_user_idea(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: COLLECT USER IDEA ===")
    state["session_id"] = state.get("session_id") or str(uuid.uuid4())
    log_timestamp(f"🆔 Session: {state['session_id']}")
    return state



def generate_questions(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: GENERATE QUESTIONS ===")
    user_text = state.get("user_raw_input", "")
    prompt = f"""
You are a Phaser mini-game UX designer.


Generate 3 concise multiple-choice questions to help design a simple and feasible  2D mini-game that can be made using phaser and llm (like dress-up, football, or board game).


For each question, include 3 options and mark if required.


Respond as JSON:
[
  {{"question":"text","options":["opt1","opt2"],"required":true}},
  ...
]


User Idea:
\"\"\"{user_text}\"\"\"
"""
    out = llm_invoke_text(prompt, state)
    try:
        state["questions"] = safe_json_parse(out)
        log_timestamp(f"✅ Generated {len(state['questions'])} questions.")
    except Exception:
        log_timestamp("⚠️ Failed to parse questions. Using fallback.")
        state["questions"] = [
            {"question": "Game theme?", "options": ["Sports", "Fantasy", "Cute", "Sci-fi"], "required": True},
            {"question": "Control style?", "options": ["Keyboard", "Mouse", "Touch"], "required": True},
        ]
    return state



def collect_user_answers(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: COLLECT USER ANSWERS ===")
    payload = {
        "session_id": state["session_id"],
        "questions": state["questions"],
        "message": "Please answer the questions below to personalize your game.",
    }
    answers = interrupt(payload)
    state["answers"] = answers or []
    log_timestamp(f"✅ Received {len(state['answers'])} answers.")
    return state



# ================================================================
# Template Selection and Customization
# ================================================================


def identify_game_template(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: IDENTIFY GAME TEMPLATE ===")
    qna = state.get("answers", [])
    user_text = state.get("user_raw_input", "")
    prompt = f"""
You are a game classifier.


Given the user's idea and Q&A, choose EXACTLY ONE template name from:
["ArenaShooter","Platformer","Runner","Fighting","Puzzle","Racing","Maze","SpaceShooter","Survival","Breakout","Sports","Idle","Camping","Dressup","BoardGame"]


User Idea:
{user_text}


Q&A:
{json.dumps(qna, indent=2)}


Respond JSON:
{{"chosen_template": "<exact_name>"}}
"""
    out = llm_invoke_text(prompt, state)
    try:
        parsed = safe_json_parse(out)
        state["chosen_template"] = parsed["chosen_template"]
        log_timestamp(f"✅ Template selected: {state['chosen_template']}")
    except Exception as e:
        log_timestamp(f"⚠️ Fallback: parsing error ({e}), defaulting to 'sports'.")
        state["chosen_template"] = "sports"
    return state



def load_template_from_library(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: LOAD TEMPLATE FROM LIBRARY ===")
    name = state["chosen_template"]
    path = f"library/{name}.html"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        state["base_template_code"] = f.read()
    log_timestamp(f"📦 Loaded template: {name}.html ({len(state['base_template_code'])} chars)")
    return state



def suggest_visual_feature_changes(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: SUGGEST VISUAL + FEATURE CHANGES ===")
    qna = state.get("answers", [])
    template = state.get("chosen_template", "")
    base_code = state.get("base_template_code", "")[:2500]


    prompt = f"""
You are customizing a mini-game that will be build using only html , css and js by a LLM model.


User Idea:
{state['user_raw_input']}


Template Type: {template}


Q&A:
{json.dumps(qna, indent=2)}


Base Game Code (snippet):
{base_code}


Suggest minor and simple and feasible specific changes to visuals and gameplay features that fit the user's preferences AND CAN BE MADE USING LLM.
tweak only what is necessary , be very controlled while making changes do not be extensive

Respond JSON only:
{{
  "Changes": ["part of code that needs to be changed/replaced and with what new code to replaced with" ]
}}
"""
    out = llm_invoke_text(prompt, state)
    try:
        mods = safe_json_parse(out)
        state["game_modifications"] = mods
        log_timestamp(f"✅ Got {len(mods.get('visual_changes', []))} visual and {len(mods.get('feature_changes', []))} feature changes.")
    except Exception:
        state["game_modifications"] = {"visual_changes": [], "feature_changes": []}
        log_timestamp("⚠️ Failed to parse mods JSON.")
    return state



def apply_changes_to_template(state: GameAgentState) -> GameAgentState:
    print("\n=== NODE: APPLY CHANGES TO TEMPLATE ===")
    changes = state.get("game_modifications", {})
    print(f"Changes are {changes}")
    base_code = state.get("base_template_code", "")
    prompt = f"""
You are a js developer also using only html and js and css.


TASK:
Apply the following modifications to this game code safely without breaking it.
do not cause overrides of funsction , scenes or anything.



{json.dumps(changes.get('Changes', []), indent=2)}




Return the COMPLETE working HTML (no markdown).

Below is The existing code , use it as a base
────────────────────────────
{base_code}
"""
    out = llm_invoke_text(prompt, state)
    html = re.sub(r"^```(?:html)?\s*", "", out.strip())
    html = re.sub(r"\s*```$", "", html)
    state["generated_code"] = html
    log_timestamp(f"✅ Code customized ({len(html)} chars)")
    return state



# ================================================================
# REVIEW + FIX + FINALIZE (reuse from your old system)
# ================================================================
def review_code(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: REVIEW GAME CODE (ENHANCED INTENT-AWARE)---")
    print("="*60)
    
    code = state.get("generated_code", "")
    engine = state.get("engine_choice", "PHASER")
    fix_iteration = state.get("fix_iteration", 0)
    intent = state.get("intent", {})
    mechanics = state.get("mechanics_blueprint", {})
    pseudocode = state.get("pseudocode_plan", "")
    
    log_timestamp(f"🔍 Starting enhanced review (iteration #{fix_iteration})...")
    log_timestamp(f"📊 Code stats: {len(code)} chars, {code.count('function')} functions")


    # Quick structure summary
    log_timestamp("🏗️ Structural analysis:")
    log_timestamp(f"   - Has <!DOCTYPE>: {code.strip().startswith('<!DOCTYPE')}")
    log_timestamp(f"   - Has <html>: {('<html' in code.lower())}")
    log_timestamp(f"   - Has <canvas>: {('<canvas' in code.lower())}")
    log_timestamp(f"   - Has <script>: {('<script' in code.lower())}")
    log_timestamp(f"   - Scene lifecycle: {'create(' in code or 'update(' in code}")
    log_timestamp(f"   - Particle references: {'particles' in code.lower()}")
    
    # Construct enhanced review prompt
    prompt = f"""
You are reviewing this HTML  game.
========================IMPORTANT========================
make sure this game is playable 
Do not worry about best practices and functions overrides , if anything does not cause runTime error then pass it

──────────────────────────────
GAME CODE 
──────────────────────────────
{code}

──────────────────────────────
RESPONSE FORMAT (STRICT JSON)
──────────────────────────────
Return a JSON object:
{{
  "status": "pass" | "fail",
  "issues": ["list of gameplay, visual, or semantic issues"],
  "suggestions": "one concise paragraph with fixes or improvements"
}}
"""


    log_timestamp("🧠 Running multi-level review (intent + visual + engine)...")
    out = llm_invoke_text(prompt, state)


    print("\n" + "-"*60)
    print("📋 RAW REVIEW RESPONSE:")
    print("-"*60)
    print(out[:800])
    if len(out) > 800:
        print(f"\n... [{len(out) - 800} more chars] ...")
    print("-"*60 + "\n")


    try:
        review = safe_json_parse(out)
        status = review.get("status", "fail")
        issues = review.get("issues", [])
        
        log_timestamp(f"📋 Review result: {status.upper()}")
        if status == "fail":
            for i, issue in enumerate(issues, 1):
                log_timestamp(f"   ❌ {i}. {issue}")
        else:
            log_timestamp("✅ Code passed all checks (intent + engine + visual)")
        
        if review.get("suggestions"):
            log_timestamp(f"💡 Suggestions: {review['suggestions'][:200]}")


    except Exception as e:
        log_timestamp(f"⚠️  Review parsing failed: {str(e)}")
        review = {
            "status": "fail",
            "issues": ["Could not parse reviewer output.", "Review response malformed."],
            "suggestions": out[:800]
        }


    state["review_notes"] = review
    return state



def fix_game_code(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: FIX GAME CODE---")
    print("="*60)
    
    review = state.get("review_notes", {}) or {}
    status = review.get("status")
    fix_iteration = state.get("fix_iteration", 0)
    
    # Increment iteration counter
    fix_iteration += 1
    state["fix_iteration"] = fix_iteration
    
    log_timestamp(f"🔧 Fix iteration #{fix_iteration}")
    
    if status == "pass":
        log_timestamp("✅ Review passed - no fixes needed")
        state["final_code"] = state.get("generated_code", "")
        return state


    issues = review.get("issues", [])
    suggestions = review.get("suggestions", "")
    code = state.get("generated_code", "")
    engine = state.get("engine_choice", "PHASER")
    
    log_timestamp(f"🔨 Attempting to fix {len(issues)} issue(s)...")
    log_timestamp(f"📝 Original code: {len(code)} chars")
    
    # ✅ Log issues being fixed
    print("\n" + "-"*60)
    print("🐛 ISSUES TO FIX:")
    print("-"*60)
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue}")
    print("-"*60)
    
    if suggestions:
        print("\n" + "-"*60)
        print("💡 SUGGESTIONS:")
        print("-"*60)
        print(suggestions[:500])
        print("-"*60 + "\n")


    # ✅ Log code being sent for fixing
    print("\n" + "-"*60)
    print("📄 ORIGINAL CODE BEING FIXED (first 1000 chars):")
    print("-"*60)
    print(code[:1000])
    if len(code) > 1000:
        print(f"\n... [{len(code) - 1000} more chars] ...")
    print("-"*60 + "\n")


    prompt = f"""
You are a game developer fixing code based on feedback.


ISSUES IDENTIFIED:
{json.dumps(issues, indent=2)}


SUGGESTIONS:
{suggestions}


FIXING PRINCIPLES:


Keep all working code unchanged. Only modify what's broken.


ORIGINAL CODE:
{code}


Return ONLY the complete fixed HTML. No markdown fences. No explanations.
Start with: <!DOCTYPE html>
"""
    
    log_timestamp("⏳ Calling LLM to fix code...")
    out = llm_invoke_text(prompt, state)
    
    # ✅ Log raw fix response
    print("\n" + "-"*60)
    print("🔧 RAW FIX RESPONSE (first 1500 chars):")
    print("-"*60)
    print(out[:1500])
    if len(out) > 1500:
        print(f"\n... [{len(out) - 1500} more chars] ...")
    print("-"*60 + "\n")


    fixed = out.strip()
    fixed = re.sub(r"^```(?:html)?\s*", "", fixed)
    fixed = re.sub(r"\s*```$", "", fixed)
    
    log_timestamp(f"✅ Fixed code received: {len(fixed)} chars")
    log_timestamp(f"📊 Size change: {len(fixed) - len(code):+d} chars")
    
    # ✅ Log what changed
    log_timestamp("🔍 Analyzing changes:")
    log_timestamp(f"   - Original functions: {code.count('function ')}")
    log_timestamp(f"   - Fixed functions: {fixed.count('function ')}")
    log_timestamp(f"   - Original event listeners: {code.count('addEventListener')}")
    log_timestamp(f"   - Fixed event listeners: {fixed.count('addEventListener')}")
    log_timestamp(f"   - Original game loops: {code.count('requestAnimationFrame') + code.count('setInterval')}")
    log_timestamp(f"   - Fixed game loops: {fixed.count('requestAnimationFrame') + fixed.count('setInterval')}")
    
    # ✅ Check if fix looks valid
    if not fixed.strip().startswith('<!DOCTYPE') and not fixed.strip().startswith('<html'):
        log_timestamp("⚠️  WARNING: Fixed code doesn't start with valid HTML!")
        log_timestamp(f"   Starts with: {fixed[:50]}")
    
    if len(fixed) < 100:
        log_timestamp("⚠️  WARNING: Fixed code suspiciously short!")
    
    # ✅ Show side-by-side comparison preview
    print("\n" + "-"*60)
    print("📊 CODE COMPARISON:")
    print("-"*60)
    print(f"ORIGINAL (first 500 chars):\n{code[:500]}\n")
    print(f"FIXED (first 500 chars):\n{fixed[:500]}\n")
    print("-"*60 + "\n")
    
    # Update both final_code and generated_code so review sees the fix
    state["final_code"] = fixed
    state["generated_code"] = fixed
    
    log_timestamp(f"🔄 Sending back to review for iteration #{fix_iteration + 1}")
    
    return state


def finalize_output(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: FINALIZE OUTPUT---")
    print("="*60)
    
    final_html = state.get("final_code") or state.get("generated_code") or ""
    fix_iteration = state.get("fix_iteration", 0)
    
    log_timestamp(f"🎉 Finalizing game after {fix_iteration} fix iteration(s)")
    log_timestamp(f"📄 Final HTML: {len(final_html)} chars")
    
    summary = {
        "session_id": state.get("session_id"),
        "engine_choice": state.get("engine_choice"),
        "engine_reasoning": state.get("engine_reasoning"),
        "design_summary": state.get("design_doc")[:200] if state.get("design_doc") else "",
        "fix_iterations": fix_iteration,
    }
    
    state["final_summary"] = "Game generation complete."
    state["final_response"] = {
        "summary": summary,
        "html": final_html,
    }
    
    log_timestamp("✅ Game generation pipeline complete!")
    print("="*60 + "\n")
    print(final_html)
    print("="*60 + "\n")


    
    return state
def collect_user_feedback(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: COLLECT USER FEEDBACK---")
    print("="*60)
    
    log_timestamp("⏸️  Waiting for user feedback on the generated game...")
    
    payload = {
        "message": "You can describe changes you'd like to make to the game (e.g., 'change background color to red', 'make player faster', etc.)",
        "session_id": state.get("session_id"),
        "previous_code_snippet": state.get("final_code", "")[:1000],
    }
    
    feedback = interrupt(payload)
    state["user_feedback"] = feedback or ""
    
    # Track iterations
    state["feedback_iteration"] = state.get("feedback_iteration", 0) + 1
    history = state.get("feedback_history", [])
    history.append({"iteration": state["feedback_iteration"], "feedback": state["user_feedback"]})
    state["feedback_history"] = history
    
    log_timestamp(f"✅ Received feedback iteration #{state['feedback_iteration']}")
    return state
def apply_feedback_to_code(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: APPLY FEEDBACK TO CODE---")
    print("="*60)
    
    feedback = state.get("user_feedback", "")
    code = state.get("final_code") or state.get("generated_code", "")
    
    if not feedback:
        log_timestamp("⚠️  No feedback provided, skipping...")
        return state
    
    log_timestamp(f"🛠️  Applying user feedback: {feedback[:120]}...")
    
    prompt = f"""
You are an expert game developer modifying an existing HTML game.


USER FEEDBACK:
"{feedback}"


TASK:
1. Locate and modify the exact part of the code affected by the feedback.
2. Make a visible or functional change (e.g., if 'make player faster', increase player velocity).
3. Keep all other code identical.
4. Return the full HTML starting with <!DOCTYPE html> (no markdown fences, no explanation).


CURRENT GAME CODE:
{code}
"""
    
    log_timestamp("⏳ Calling OpenAI API...")
    out = llm_invoke_text(prompt, state)
    log_timestamp("✅ OpenAI API responded")


    # Extract HTML from response
    fixed = re.sub(r"^```(?:html)?\s*", "", out)  # remove leading ``` or ```html
    fixed = re.sub(r"\s*```$", "", fixed)         # remove trailing ```
    # ✅ Check if change was actually applied
    if fixed.strip() == code.strip():
        log_timestamp("⚠️  No visible change detected. Reinforcing feedback...")
        
        # ✅ FIX: All code using stronger_prompt must be INSIDE the if block
        stronger_prompt = f"""
Forcefully apply the feedback below and ensure visible or functional difference.


USER FEEDBACK (MANDATORY TO IMPLEMENT):
{feedback}


CURRENT CODE:
{code}


Return ONLY the modified HTML. No markdown. Start with <!DOCTYPE html>
"""
        log_timestamp("⏳ Calling OpenAI API with stronger prompt...")
        out = llm_invoke_text(stronger_prompt)
        log_timestamp("✅ OpenAI API responded")


        # Extract HTML from response
        fixed = re.sub(r"^```(?:html)?\s*", "", out)  # remove leading ``` or ```html
        fixed = re.sub(r"\s*```$", "", fixed)         # remove trailing ```
    
    # Update state with fixed code
    state["final_code"] = fixed
    state["generated_code"] = fixed  # ✅ Also update for consistency
    
    log_timestamp(f"✅ Feedback applied, new code length: {len(fixed)} chars")
    log_timestamp(f"📊 Code change: {len(fixed) - len(code):+d} chars")
    
    return state


def verify_feedback_applied(state: GameAgentState) -> GameAgentState:
    feedback = state.get("user_feedback", "")
    code = state.get("final_code", "")
    if not feedback or not code:
        return state


    prompt = f"""
You are verifying if this feedback has been implemented.


Feedback: "{feedback}"
Code: {code}


Respond JSON only:
{{ "implemented": true|false, "evidence": "brief reason" }}
"""
    out = llm_invoke_text(prompt, state)
    try:
        result = safe_json_parse(out)
        if not result.get("implemented", True):
            log_timestamp("⚠️ Feedback not applied. Reapplying once...")
            state = apply_feedback_to_code(state)
    except Exception:
        log_timestamp("⚠️ Feedback verification failed, skipping.")
    return state




# ================================================================
# WORKFLOW BUILD
# ================================================================
memory = MemorySaver()
workflow = StateGraph(GameAgentState)


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


workflow.set_entry_point("collect_user_idea")


workflow.add_edge("collect_user_idea", "generate_questions")
workflow.add_edge("generate_questions", "collect_user_answers")
workflow.add_edge("collect_user_answers", "identify_game_template")
workflow.add_edge("identify_game_template", "load_template_from_library")
workflow.add_edge("load_template_from_library", "suggest_visual_feature_changes")
workflow.add_edge("suggest_visual_feature_changes", "apply_changes_to_template")
workflow.add_edge("apply_changes_to_template", "review_code")


def review_branch(state: GameAgentState): 
    rn = state.get("review_notes") or {}
    status = rn.get("status", "fail")
    next_node = "fix_game_code" if status == "fail" else "finalize_output"
    
    log_timestamp(f"🔀 Branch decision: {status} → {next_node}")
    
    return next_node


workflow.add_conditional_edges("review_code", review_branch)

def fix_branch(state: GameAgentState) -> str:
    fix_iteration = state.get("fix_iteration", 0)
    
    if fix_iteration > MAX_FIX_ITERATIONS:  # ✅ Uses constant
        log_timestamp(f"🚫 Max fix iterations ({MAX_FIX_ITERATIONS}) reached")  # ✅ Fixed syntax
        return "finalize_output"
    
    return "review_code"

workflow.add_conditional_edges("fix_game_code", fix_branch) 
workflow.add_edge("finalize_output", END)
workflow.add_edge("collect_user_feedback", "apply_feedback_to_code")
workflow.add_edge("apply_feedback_to_code", "verify_feedback_applied")

def feedback_loop_decision(state: GameAgentState) -> str:
    """
    Decide whether to end the feedback loop or continue.
    
    Returns:
        END: Exit workflow (return to user)
        str: Node name to continue to (e.g., "collect_user_feedback")
    """
    feedback_iteration = state.get("feedback_iteration", 0)
    
    log_timestamp(f"📊 Feedback iteration #{feedback_iteration} complete")
    
    # Check if we've reached max feedback iterations
    if feedback_iteration >= MAX_FEEDBACK_ITERATIONS:
        log_timestamp(f"✅ Reached max feedback iterations ({MAX_FEEDBACK_ITERATIONS}), ending workflow")
        return END
    
    # Default: End (one feedback round only)
    # To enable continuous feedback, change to: return "collect_user_feedback"
    return END

workflow.add_conditional_edges("verify_feedback_applied", feedback_loop_decision)


game_agent_app = workflow.compile(checkpointer=memory)

