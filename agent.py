
import os
import json
import uuid
import re
from typing import TypedDict, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running.")

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

# Gemini LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize LLM (Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")


# -------------------------
# State type definition
# -------------------------
class GameAgentState(TypedDict, total=False):
    session_id: str
    user_raw_input: str
    intent: Dict[str, Any]
    questions: List[Dict[str, Any]]
    answers: List[Dict[str, Any]]
    validated: bool
    design_doc: str
    design_doc_structured: Dict[str, Any]
    engine_choice: str
    engine_reasoning: str
    game_prompt: str
    generated_code: str
    review_notes: Dict[str, Any]
    final_code: str
    final_summary: str
    final_response: Dict[str, Any]
    fix_iteration: int  # Track fix iterations
    user_feedback: str
    feedback_iteration: int
    feedback_history: List[Dict[str, str]]



# -------------------------
# Utility helpers
# -------------------------
def log_timestamp(message: str):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def safe_json_parse(s: str):
    """
    Try to find and parse the first JSON object in string s.
    Returns Python object or raises ValueError.
    """
    # Quick try full string
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to locate JSON-like substring
    m = re.search(r'(\{[\s\S]*\})', s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try array case
    m2 = re.search(r'(\[[\s\S]*\])', s)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            pass

    raise ValueError("No JSON found in string")


def llm_invoke_text(prompt: str) -> str:
    """Invoke Gemini and return the raw textual content (string)."""
    log_timestamp("🔄 Calling Gemini API...")
    start_time = datetime.now()
    
    resp = llm.invoke(prompt)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log_timestamp(f"✅ Gemini API responded in {elapsed:.2f}s")
    
    # Depending on the wrapper, resp may have .content or .text; try both
    content = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
    return content.strip()


# -------------------------
# Node implementations
# -------------------------
def intent_analysis(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: INTENT ANALYSIS---")
    print("="*60)
    
    user_text = (state.get("user_raw_input") or "").strip()
    if not user_text:
        log_timestamp("⚠️  No user input found, skipping intent analysis")
        state["intent"] = {}
        return state

    log_timestamp(f"📝 Analyzing user input: {user_text[:100]}...")
    
    prompt = f"""
You are an expert game design analyst. Extract the user's intent from the following freeform input.
Respond ONLY with a JSON object exactly in this shape:

{{
  "genre": "one-word genre (shooter, platformer, puzzle, runner, puzzle, simulation, etc.)",
  "player_goal": "short description of the player's main goal",
  "key_entities": ["list","of","important","entities"],
  "aesthetic_vibe": ["keywords","like","dark","cute","neon","retro"],
  "summary": "one-sentence pitch summarizing the concept"
}}

User input:
\"\"\"{user_text}\"\"\"
"""
    out = llm_invoke_text(prompt)
    try:
        parsed = safe_json_parse(out)
        log_timestamp(f"✅ Intent extracted: {parsed.get('genre', 'unknown')} game")
    except Exception as e:
        log_timestamp(f"⚠️  Intent parsing failed: {str(e)}, using fallback")
        parsed = {
            "genre": "unknown",
            "player_goal": "",
            "key_entities": [],
            "aesthetic_vibe": [],
            "summary": user_text,
        }
    state["intent"] = parsed
    return state


def collect_user_idea(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: COLLECT USER IDEA---")
    print("="*60)
    
    if not state.get("session_id"):
        state["session_id"] = str(uuid.uuid4())
        log_timestamp(f"🆔 Generated session ID: {state['session_id']}")
    else:
        log_timestamp(f"🆔 Using existing session ID: {state['session_id']}")
    
    state["user_raw_input"] = state.get("user_raw_input", "")
    state["validated"] = False
    state["fix_iteration"] = 0  # Initialize fix counter
    
    return state


def generate_questions(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: GENERATE QUESTIONS---")
    print("="*60)
    
    user_text = state.get("user_raw_input", "").strip()
    intent = state.get("intent") or {}
    
    log_timestamp(f"🎯 Generating questions based on intent: {intent.get('summary', 'N/A')[:80]}...")

    prompt = f"""
You are a senior game designer and UX writer building mini games for phaserjs. The user idea: {json.dumps(user_text)}

Use the user's intent metadata (if available) to craft 6 clarifying questions that will get the information needed to author a aimple 2D playable game.

For each question provide 3-5 multiple-choice options and mark whether the question is critical ("required": true/false).

Respond ONLY with JSON array of objects:
[
  {{
    "question": "text",
    "options": ["opt1", "opt2", "..."],
    "required": true
  }},
  ...
]

If you need to assume defaults, still provide sensible options.
"""
    out = llm_invoke_text(prompt)
    try:
        questions = safe_json_parse(out)
        log_timestamp(f"✅ Generated {len(questions)} questions")
    except Exception as e:
        log_timestamp(f"⚠️  Question generation failed: {str(e)}, using fallback")
        questions = [
            {"question": "Preferred weapon or attack type?", "options": ["Laser", "Bullet", "Fireball"], "required": True},
            {"question": "Environment / background vibe?", "options": ["Forest", "City", "Space"], "required": True},
            {"question": "Target type?", "options": ["Monsters", "Bottles", "Aliens"], "required": True},
            {"question": "Desired difficulty?", "options": ["Easy", "Normal", "Hard"], "required": False},
            {"question": "Prefer single-screen or levels?", "options": ["Single-screen", "Multiple levels"], "required": False},
            {"question": "Any must-have mechanics?", "options": ["Jumping", "Shooting", "Dodging"], "required": False},
        ]
    state["questions"] = questions
    return state


def collect_user_answers(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: COLLECT USER ANSWERS (interrupt to human)---")
    print("="*60)
    
    log_timestamp("⏸️  Interrupting for human input...")
    
    payload = {
        "session_id": state.get("session_id"),
        "questions": state.get("questions", []),
        "message": "Please answer these questions (choose or type custom answers) to personalize your game."
    }
    
    answers = interrupt(payload)
    state["answers"] = answers or []
    
    log_timestamp(f"✅ Received {len(state['answers'])} answers from user")
    
    return state


def validate_inputs(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: VALIDATE INPUTS---")
    print("="*60)
    
    missing = []
    if not state.get("user_raw_input"):
        missing.append("user_raw_input")
    if not state.get("answers"):
        missing.append("answers")
    
    if missing:
        log_timestamp(f"❌ Validation failed: Missing {', '.join(missing)}")
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")
    
    log_timestamp("✅ All inputs validated successfully")
    state["validated"] = True
    return state

def generate_mechanics_blueprint(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: GENERATE MECHANICS BLUEPRINT (ENHANCED)---")
    print("="*60)
    
    user_text = state.get("user_raw_input", "")
    intent = state.get("intent", {})
    answers = state.get("answers", [])
    questions = state.get("questions", [])
    design_struct = state.get("design_doc_structured", {})

    qa_pairs = []
    for i, ans in enumerate(answers):
        q_text = questions[i]["question"] if i < len(questions) else "Unknown question"
        qa_pairs.append(f"{i+1}. {q_text} → {ans.get('answer', '')}")
    qa_combined = "\n".join(qa_pairs)

    log_timestamp("🧠 Generating detailed mechanics + art blueprint...")
    print(qa_combined)

    prompt = f"""
You are a senior **game systems designer and art director** working together.
Using the following information, generate a **comprehensive pseudocode-style blueprint** for both the mechanics and the presentation of a 2D game.

────────────────────────
User Idea:
{user_text}

Intent Metadata:
{json.dumps(intent, indent=2)}

Structured Design Summary:
{json.dumps(design_struct, indent=2)}

User's Demand:
{json.dumps(qa_combined, indent=2)}
────────────────────────

Your job is to imagine the game as if it were in early prototyping — describe all entities, mechanics, and artistic presentation precisely enough for a developer to implement them in Phaser or HTML Canvas.

────────────────────────
OUTPUT FORMAT (STRICT JSON ONLY)
────────────────────────
Return JSON with the following structure:

{{
  "entities": [
    {{
      "name": "player",
      "role": "main controllable entity",
      "behavior": "e.g., moves with WASD, fires projectiles, takes damage on collision",
      "variables": ["x", "y", "velocity", "health", "score"],
      "animations": ["idle", "move", "attack", "hit"],
      "visual_style": {{
        "shape": "e.g., humanoid, blob, jellyfish, robot",
        "colors": ["hex1", "hex2"],
        "effects": ["glow", "trail", "pulsing", "outline"]
      }},
      "sound_effects": ["jump", "shoot", "damage"]
    }},
    {{
      "name": "enemy",
      "role": "hostile or obstacle",
      "behavior": "e.g., patrols, chases, fires projectiles, explodes",
      "variables": ["x", "y", "velocity", "hp"],
      "animations": ["spawn", "attack", "death"],
      "visual_style": {{
        "shape": "e.g., octopus, alien, rock monster",
        "colors": ["#ff3300", "#aa0000"],
        "effects": ["smoke", "pulse", "glow"]
      }},
      "sound_effects": ["attack", "death"]
    }},
    {{
      "name": "projectile",
      "role": "object fired by player or enemy",
      "behavior": "travels in straight line, collides, disappears",
      "visual_style": {{
        "shape": "orb",
        "colors": ["#00ffff"],
        "effects": ["trail", "flash"]
      }}
    }}
  ],
  "environment": {{
    "theme": "e.g., neon city, coral reef, alien desert",
    "lighting": "day|night|neon|underwater|dynamic",
    "background_elements": ["mountains", "stars", "bubbles", "ruins"],
    "foreground_effects": ["fog", "particles", "light rays"],
    "color_palette": ["#hex", "#hex", "#hex"],
    "music_style": "e.g., ambient synthwave, upbeat chiptune, dark orchestral"
  }},
  "core_loop": [
    "spawn enemies periodically or per wave",
    "player moves and performs actions",
    "handle collisions (projectiles vs enemies, player vs enemy)",
    "update health, score, and level progression",
    "check win and lose conditions"
  ],
  "controls": ["list of control inputs like WASD, SPACE, MOUSE"],
  "special_mechanics": ["e.g., slow motion, shield, combo meter, tentacle attack"],
  "ui": {{
    "elements": ["score", "health", "level", "timer"],
    "layout": "top-left for score, top-right for health, bottom for level",
    "style": "pixel font | neon digital | minimal white text"
  }},
  "camera_behavior": "fixed | follow player | scrolling | zoom",
  "win_condition": "explicit condition",
  "lose_condition": "explicit condition",
  "estimated_complexity": "low|medium|high"
}}

────────────────────────
Guidelines:
────────────────────────
- Combine **mechanical logic + artistic vision**.
- Be descriptive yet implementable.
- Avoid generic phrases like “looks good”; define actual shapes, colors, or moods.
- Match art style to the intent’s vibe (dark, retro, cute, neon, fantasy, etc.).
- Use consistent terminology and JSON syntax.
- Include at least one unique or signature mechanic that makes the game memorable.
"""

    out = llm_invoke_text(prompt)

    try:
        blueprint = safe_json_parse(out)
        log_timestamp(f"✅ Mechanics + Art blueprint parsed successfully with {len(blueprint.get('entities', []))} entities.")
        log_timestamp(f"🎨 Art theme: {blueprint.get('environment', {}).get('theme', 'unknown')}")
        log_timestamp(f"📊 Estimated complexity: {blueprint.get('estimated_complexity', 'unknown')}")
    except Exception as e:
        log_timestamp(f"⚠️ Blueprint parsing failed: {str(e)}")
        log_timestamp(f"⚠️ Raw output (first 500 chars): {out[:500]}")
        blueprint = {
            "entities": [],
            "core_loop": [],
            "controls": [],
            "environment": {},
            "ui": {},
            "special_mechanics": [],
            "estimated_complexity": "medium",
            "note": "Failed to parse JSON output"
        }

    state["mechanics_blueprint"] = blueprint
    return state

def complexity_branch(state: GameAgentState):
    """
    Branches based on the game's estimated complexity.
    Routes to pseudocode generation for medium/high complexity,
    or directly to prompt build for low complexity.
    """
    print("\n" + "="*60)
    print("---NODE: COMPLEXITY BRANCH---")
    print("="*60)
    
    # Try to detect complexity from structured blueprint or mechanics
    design_struct = state.get("design_doc_structured", {})
    mechanics = state.get("mechanics_blueprint", {})
    est_complexity = (
        mechanics.get("estimated_complexity") or 
        design_struct.get("estimated_complexity") or 
        "medium"
    ).lower()
    
    log_timestamp(f"🧠 Detected game complexity: {est_complexity}")
    
    if est_complexity in ["low", "simple"]:
        log_timestamp("⚡ Simple game — skipping pseudocode generation.")
        next_node = "build_code_prompt"
    else:
        log_timestamp("🧩 Medium/High complexity — generating pseudocode plan.")
        next_node = "generate_pseudocode_plan"
    
    return next_node

def generate_pseudocode_plan(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: GENERATE PSEUDOCODE PLAN (BALANCED)---")
    print("="*60)
    
    design_struct = state.get("design_doc_structured", {})
    mech_blueprint = state.get("mechanics_blueprint", {})
    user_text = state.get("user_raw_input", "")
    intent = state.get("intent", {})

    log_timestamp("🧩 Generating human-readable pseudocode plan...")

    prompt = f"""
You are a senior 2D phaserJS game developer planning the gameplay logic before implementation.

Your task is to write a **clear, structured pseudocode plan** — like an annotated script — that explains how the game should work, step by step.  
Use indentation, comments, and consistent formatting.  
The pseudocode should feel like real development planning notes, not JSON.

──────────────────────────────
GAME DESIGN INPUTS
──────────────────────────────

Mechanics Blueprint:
{json.dumps(mech_blueprint, indent=2)}

──────────────────────────────
OUTPUT FORMAT
──────────────────────────────
Write plain text pseudocode with the following sections (use clear headers):

SETUP:
- Game window, engine initialization, physics setup
- Asset placeholders (procedural shapes if mentioned)

ENTITIES:
- Describe each entity (player, enemies, projectiles)
- Include attributes and behaviors
- Mention key variables and interactions

CONTROLS:
- Describe control mappings and how player movement/actions work on top left corner

GAME LOOP:
- Describe the frame update process
- Movement, collisions, AI, and level updates

COLLISIONS:
- Define collision outcomes (damage, scoring, destruction)

UI + FEEDBACK:
- HUD, health, score, level, effects, visual or sound feedback

SPECIAL MECHANICS:
- Mention unique or creative systems (e.g., time freeze, tentacle shooting, portals)

PROGRESSION:
- Explain how the difficulty or level changes

WIN / LOSE CONDITIONS:
- Define victory and defeat clearly

ART + AUDIO NOTES:
- Mention colors, style, lighting, animations, and ambient music direction

──────────────────────────────
Guidelines:
──────────────────────────────
- Be descriptive, but concise.
- Use “//” comments liberally to annotate reasoning.
- Write as if another developer will implement it.
- Keep indentation consistent.
- Avoid JSON or natural-language paragraphs.
- Example:
    SETUP:
        // Initialize Phaser game with 800x600 window
        // Enable Arcade physics, gravity = 0
    ENTITIES:
        player:
            // Moves with WASD, shoots with SPACE
            // Has 100 HP, 3 lives
        enemy:
            // Spawns at random edges, chases player
            // Explodes on hit
"""

    out = llm_invoke_text(prompt)

    # Try to clean or verify it’s not blank
    cleaned = out.strip()
    if not cleaned or len(cleaned) < 50:
        log_timestamp("⚠️ LLM returned too short pseudocode, using fallback.")
        cleaned = (
            "SETUP:\n"
            "    // Initialize basic 2D game window and physics\n"
            "ENTITIES:\n"
            "    // Player moves with WASD and shoots projectiles\n"
            "    // Enemies spawn periodically and move toward player\n"
            "GAME LOOP:\n"
            "    // Handle input, update entities, detect collisions\n"
            "    // Display score and health\n"
        )

    # Store in state
    state["pseudocode_plan"] = cleaned
    log_timestamp(f"✅ Pseudocode plan generated ({len(cleaned)} chars)")
    log_timestamp("📘 Sample preview:")
    print("\n" + cleaned[:400] + ("\n... (truncated)" if len(cleaned) > 400 else ""))
    
    return state


def build_code_prompt(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: BUILD CODE PROMPT (ADAPTIVE PHASER VERSION)---")
    print("="*60)
    
    # Retrieve context
    title = state.get("design_doc_structured", {}).get("title", "Untitled Game")
    genre = state.get("design_doc_structured", {}).get("genre", "arcade")
    design_summary = state.get("design_doc", "")
    design_struct = state.get("design_doc_structured", {})
    mechanics = state.get("mechanics_blueprint", {})
    pseudocode = state.get("pseudocode_plan", "")
    complexity = (
        mechanics.get("estimated_complexity")
        or design_struct.get("estimated_complexity")
        or "medium"
    ).lower()

    # Environment and art details
    environment = mechanics.get("environment", {})
    theme = environment.get("theme", "default")
    lighting = environment.get("lighting", "standard")
    color_palette = environment.get("color_palette", ["#000000", "#FFFFFF"])
    art_notes = design_struct.get("art_notes", mechanics.get("art_notes", ""))

    # Controls and style
    controls = design_struct.get("controls", mechanics.get("controls", ["WASD"]))
    control_text = ", ".join(controls) if isinstance(controls, list) else controls

    log_timestamp(f"📝 Building Phaser code generation prompt for: {title} ({complexity} complexity)")

    # ======================================================
    # PROMPT START
    # ======================================================
    prompt = f"""
You are an expert **Phaser 3** game developer.
Generate a **fully functional**, playable HTML game using Phaser 3 based on the following data.

Your goal:
- Create a self-contained, bug-free Phaser 3 game.
- Reflect the mechanics, pseudocode, and art/environment styles provided.
- The game must load and run directly in the browser with no external assets.
- Use procedural shapes for all sprites and effects and make sure they look like characters (have hands , eyes etc) and not just (squares , triangle , shapes etc)

═══════════════════════════════════════════════════════════
🎮 GAME CONTEXT
═══════════════════════════════════════════════════════════
Title: {title}
Genre: {genre}
Complexity: {complexity}
Theme: {theme}
Lighting: {lighting}
Color Palette: {color_palette}
Controls: {control_text}

═══════════════════════════════════════════════════════════
⚙️ MECHANICS BLUEPRINT
═══════════════════════════════════════════════════════════
{json.dumps(mechanics, indent=2)}

═══════════════════════════════════════════════════════════
📜 IMPLEMENTATION PLAN (PSEUDOCODE)
═══════════════════════════════════════════════════════════
{pseudocode or "// No pseudocode generated (simple game)."}

═══════════════════════════════════════════════════════════
🎨 ART + ENVIRONMENT NOTES
═══════════════════════════════════════════════════════════
{art_notes or "// Use a visual style that matches theme and genre."}

═══════════════════════════════════════════════════════════
💡 IMPLEMENTATION INSTRUCTIONS
═══════════════════════════════════════════════════════════
Follow these rules when generating the code:

1. **Use Phaser 3** only.
2. No external files, images, or assets — draw entities using Phaser's graphics API.
3. Implement all entities, collisions, and behaviors mentioned in blueprint.
4. Background and lighting should visually match the theme.
5. Include proper UI (score, health, etc.) in the top corners.
9. Include a restart mechanism on game over.


═══════════════════════════════════════════════════════════
📄 OUTPUT REQUIREMENT
═══════════════════════════════════════════════════════════
Output ONLY the **final working HTML file**, starting with:
<!DOCTYPE html>

❌ Do NOT include Markdown code fences (```)
❌ Do NOT add explanations before or after
✅ Include both <style> and <script> sections

═══════════════════════════════════════════════════════════
⚙️ HINTS FOR CODE STRUCTURE
═══════════════════════════════════════════════════════════
- Use a config object and define a Phaser.Scene with preload(), create(), and update().
- Declare player, enemies, projectiles, and groups globally.
- Implement movement, firing, and collision systems.
- Add text UI for score, health, and level.
- Implement gameOver() and restart logic.

═══════════════════════════════════════════════════════════
🚀 READY TO GENERATE
═══════════════════════════════════════════════════════════
Generate the complete Phaser 3 HTML code below.
"""

    # ======================================================
    # STORE IN STATE
    # ======================================================
    state["game_prompt"] = prompt

    log_timestamp(f"✅ Code generation prompt built successfully ({len(prompt)} chars)")
    if complexity in ["low", "simple"]:
        log_timestamp("⚡ Note: Direct generation path (no pseudocode used).")
    else:
        log_timestamp("🧩 Note: Pseudocode-assisted path for complex game.")
    
    return state




def generate_game_code(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: GENERATE GAME CODE---")
    print("="*60)
    
    prompt = state.get("game_prompt", "")
    if not prompt:
        log_timestamp("❌ No game prompt available")
        raise ValueError("No game prompt available for code generation.")
    
    log_timestamp(f"🎨 Generating game code... (prompt: {len(prompt)} chars)")
    log_timestamp("⏳ This may take 30-120 seconds depending on complexity...")
    
    out = llm_invoke_text(prompt)
    
    # Extract HTML
    html_candidate = out.strip()
    html_candidate = re.sub(r"^```(?:html)?\s*", "", html_candidate)
    html_candidate = re.sub(r"\s*```$", "", html_candidate)
    state["generated_code"] = html_candidate
    

    
    log_timestamp(f"✅ Code generated: {len(html_candidate)} chars, {html_candidate.count('<')} HTML tags")
    log_timestamp(f"📄 Contains <canvas>: {('<canvas' in html_candidate.lower())}")
    log_timestamp(f"📄 Contains <script>: {('<script' in html_candidate.lower())}")
    print(f"=========html_candidate is : {html_candidate}=========")
    return state
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
You are reviewing this HTML Phaser 3 game.
========================IMPORTANT========================
make sure this game is playable and correct phaser API are used


──────────────────────────────
MECHANICS BLUEPRINT (SUMMARY)
──────────────────────────────
{json.dumps(mechanics, indent=2)}

──────────────────────────────
GAME CODE 
──────────────────────────────
{code}

──────────────────────────────
REVIEW CHECKLIST
──────────────────────────────


2. **Engine Sanity**
   eg:-
   - Any invalid Phaser API calls? (e.g., createCanvas().draw())
   - Missing generateTexture() before sprite use?
   - Proper use of preload(), create(), and update()?


3. **Visual Sanity**
   eg:-
   - Would something visible appear when loaded? (background, entities, text)
   - Are key sprites drawn or filled with color?

4. **Core Mechanics**
   eg:-
   - Is the player controllable (e.g., WASD or arrow keys)?
   - Is there a visible feedback loop (score, health)?
   - Do collisions or interactions occur?

5. **Functional Stability**
   eg:-
   - Any logic that could crash or hang (undefined vars, missing textures)?
   - GameOver / Restart systems functioning?

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
    out = llm_invoke_text(prompt)

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
You are a game developer fixing code based on QA feedback.

ENGINE: {engine}

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
    out = llm_invoke_text(prompt)
    
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
You are an expert game developer modifying an existing Phaser/HTML game.

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
    
    log_timestamp("⏳ Calling Gemini API...")
    out = llm_invoke_text(prompt)
    log_timestamp("✅ Gemini API responded")

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
        log_timestamp("⏳ Calling Gemini API with stronger prompt...")
        out = llm_invoke_text(prompt)
        log_timestamp("✅ Gemini API responded")

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
    out = llm_invoke_text(prompt)
    try:
        result = safe_json_parse(out)
        if not result.get("implemented", True):
            log_timestamp("⚠️ Feedback not applied. Reapplying once...")
            state = apply_feedback_to_code(state)
    except Exception:
        log_timestamp("⚠️ Feedback verification failed, skipping.")
    return state


# -------------------------
# Build & compile the Graph
# -------------------------
memory = MemorySaver()
workflow = StateGraph(GameAgentState)

# Add nodes (in the specified flow)
workflow.add_node("intent_analysis", intent_analysis)
workflow.add_node("collect_user_idea", collect_user_idea)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("collect_user_answers", collect_user_answers)
workflow.add_node("validate_inputs", validate_inputs)
workflow.add_node("generate_mechanics_blueprint", generate_mechanics_blueprint)
workflow.add_node("build_code_prompt", build_code_prompt)
workflow.add_node("generate_game_code", generate_game_code)
workflow.add_node("review_code", review_code)
workflow.add_node("fix_game_code", fix_game_code)
workflow.add_node("finalize_output", finalize_output)
workflow.add_node("collect_user_feedback", collect_user_feedback)
workflow.add_node("apply_feedback_to_code", apply_feedback_to_code)
workflow.add_node("verify_feedback_applied", verify_feedback_applied)
workflow.add_node("complexity_branch", complexity_branch)
workflow.add_node("generate_pseudocode_plan", generate_pseudocode_plan)

# Entry point
workflow.set_entry_point("intent_analysis")

# Edges (linear flow + conditional review->fix loop)
workflow.add_edge("intent_analysis", "collect_user_idea")
workflow.add_edge("collect_user_idea", "generate_questions")
workflow.add_edge("generate_questions", "collect_user_answers")
workflow.add_edge("collect_user_answers", "validate_inputs")
workflow.add_edge("validate_inputs", "generate_mechanics_blueprint")
workflow.add_conditional_edges("generate_mechanics_blueprint", complexity_branch)
workflow.add_edge("generate_pseudocode_plan", "build_code_prompt")
workflow.add_edge("build_code_prompt", "generate_game_code")
workflow.add_edge("generate_game_code", "review_code")
workflow.add_edge("collect_user_feedback", "apply_feedback_to_code")
workflow.add_edge("apply_feedback_to_code", "verify_feedback_applied")
workflow.add_edge("verify_feedback_applied", "review_code")

# Conditional branching based on complexity


# conditional: if review fail -> fix_game_code else finalize_output
def review_branch(state: GameAgentState):
    rn = state.get("review_notes") or {}
    status = rn.get("status", "fail")
    next_node = "fix_game_code" if status == "fail" else "finalize_output"
    
    log_timestamp(f"🔀 Branch decision: {status} → {next_node}")
    
    return next_node

workflow.add_conditional_edges("review_code", review_branch)
workflow.add_edge("fix_game_code", "review_code")
workflow.add_edge("finalize_output", END)

# compile
game_agent_app = workflow.compile(checkpointer=memory)

