
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
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


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
You are a senior game designer and UX writer. The user idea: {json.dumps(user_text)}

Use the user's intent metadata (if available) to craft 6 clarifying questions that will get the information needed to author a high-quality 2D playable game.

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


def design_game_blueprint(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: DESIGN GAME BLUEPRINT---")
    print("="*60)
    
    user_text = state.get("user_raw_input", "")
    intent = state.get("intent", {})
    answers = state.get("answers", [])
    
    log_timestamp("📐 Creating game design document...")

    prompt = f"""
You are a senior game designer. Using the information below, produce a clear, structured design document for a production-quality 2D game.
Include: objective, win/lose conditions, core mechanics, controls, progression (levels or waves), enemy behavior, scoring, UI elements (health/score), and art/style notes.

User idea:
{user_text}

Intent metadata:
{json.dumps(intent, indent=2)}

User answers (list):
{json.dumps(answers, indent=2)}

Return as two parts separated by a line '---STRUCTURED-JSON---' where the first part is a short human-readable design summary (1-3 paragraphs) and the second part is a JSON object named "design" with keys:
{{
  "title": "...",
  "objective": "...",
  "win_condition": "...",
  "lose_condition": "...",
  "core_mechanics": ["..."],
  "controls": ["..."],
  "progression": "...",
  "enemy_behavior": "...",
  "ui": {{ "score": true, "health": true }},
  "art_notes": "...",
  "estimated_complexity": "low|medium|high"
}}

Respond with the combined text exactly as specified.
"""
    out = llm_invoke_text(prompt)
    
    if '---STRUCTURED-JSON---' in out:
        human_part, json_part = out.split('---STRUCTURED-JSON---', 1)
        state["design_doc"] = human_part.strip()
        try:
            design_json = safe_json_parse(json_part.strip())
            state["design_doc_structured"] = design_json
            log_timestamp(f"✅ Design created: {design_json.get('title', 'Untitled')}")
            log_timestamp(f"📊 Complexity: {design_json.get('estimated_complexity', 'unknown')}")
        except Exception as e:
            log_timestamp(f"⚠️  Structured JSON parsing failed: {str(e)}")
            state["design_doc_structured"] = {"estimated_complexity": "medium", "note": "could not parse structured JSON"}
    else:
        log_timestamp("⚠️  No structured separator found, using full text")
        state["design_doc"] = out
        state["design_doc_structured"] = {"estimated_complexity": "medium"}

    return state


def engine_decision(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: ENGINE DECISION (LLM-driven)---")
    print("="*60)
    
    design_struct = state.get("design_doc_structured") or state.get("design_doc") or ""
    
    log_timestamp("🎮 Deciding game engine...")
    
    prompt = f"""
You are an expert 2D game engine architect. Given the following structured design information, decide whether the implementation
should use PHASER (Phaser 3) or HTML_CANVAS (vanilla Canvas/DOM). Consider complexity, physics, number of levels, animations, and performance.

Return ONLY a JSON object exactly like:
{{ "engine_choice": "PHASER" | "HTML_CANVAS", "reasoning": "one or two sentence explanation" }}

Design info:
{json.dumps(design_struct, indent=2)}
"""
    out = llm_invoke_text(prompt)
    try:
        decision = safe_json_parse(out)
        state["engine_choice"] = decision.get("engine_choice", "HTML_CANVAS")
        state["engine_reasoning"] = decision.get("reasoning", "")
        log_timestamp(f"✅ Engine selected: {state['engine_choice']}")
        log_timestamp(f"💡 Reasoning: {state['engine_reasoning']}")
    except Exception as e:
        log_timestamp(f"⚠️  Engine decision parsing failed: {str(e)}, using fallback")
        est = (design_struct.get("estimated_complexity") if isinstance(design_struct, dict) else None) or "medium"
        if est == "high":
            state["engine_choice"] = "PHASER"
            state["engine_reasoning"] = "Estimated complexity high; defaulting to Phaser."
        else:
            state["engine_choice"] = "HTML_CANVAS"
            state["engine_reasoning"] = "Estimated complexity not high; using Canvas for simplicity."
        log_timestamp(f"✅ Fallback engine: {state['engine_choice']}")
    
    return state

def build_code_prompt(state: GameAgentState) -> GameAgentState:
    print("\n" + "="*60)
    print("---NODE: BUILD CODE PROMPT---")
    print("="*60)
    
    engine = state.get("engine_choice", "HTML_CANVAS")
    design_summary = state.get("design_doc", "")
    design_struct = state.get("design_doc_structured", {})
    controls = design_struct.get("controls", ["WASD"])
    genre = design_struct.get("genre", "shooter")
    
    log_timestamp(f"📝 Building code generation prompt for {engine}...")

    prompt = f"""
You are an expert game developer. You will adapt a WORKING template to create a bug-free game.

═══════════════════════════════════════════════════════════
GAME SPECIFICATION
═══════════════════════════════════════════════════════════

Title: {design_struct.get('title', 'Game')}
Genre: {genre}
Engine: {engine}

Design Summary:
{design_summary}

Full Requirements:
{json.dumps(design_struct, indent=2)}

═══════════════════════════════════════════════════════════
WORKING TEMPLATE - COPY THIS STRUCTURE and IMPROVISE ON ONLY VISUALS OF CHARACTERS AND BG , DO NOT ALTER GAME LOGIC 
═══════════════════════════════════════════════════════════

<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{design_struct.get('title', 'Game')}</title>
<style>
body {{ margin: 0; padding: 0; background: #000; display: flex; justify-content: center; align-items: center; height: 100vh; }}
#game {{ border: 2px solid #333; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/phaser@3.55.2/dist/phaser.min.js"></script>
</head>
<body>
<div id="game"></div>
<script>

// ═══════════════════════════════════════════════════════════
// GLOBAL VARIABLES (Always declare these)
// ═══════════════════════════════════════════════════════════
var gameScene;  // Scene reference for callbacks
var player;     // Main player entity
var entities;   // Group for enemies/obstacles/collectibles
var projectiles; // Group for bullets/thrown objects

// Game state
var score = 0;
var health = 100;
var level = 1;
var gameOver = false;

// UI elements
var scoreText, healthText, levelText, instructionsText;

// Input
var keys;  // Keyboard input object

// ═══════════════════════════════════════════════════════════
// PHASER CONFIGURATION
// ═══════════════════════════════════════════════════════════
var config = {{
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    parent: 'game',
    physics: {{
        default: 'arcade',
        arcade: {{
            gravity: {{ y: 0 }},  // ADAPT: Set to 300+ for platformers, 0 for top-down
            debug: false
        }}
    }},
    scene: {{
        preload: preload,
        create: create,
        update: update
    }}
}};

var game = new Phaser.Game(config);

// ═══════════════════════════════════════════════════════════
// PRELOAD: Create Visible Sprites  (I HAVE SHARED AN EXAMPLE , You can be more creative and generate more types as needed , be pretty+professional ) 
// ═══════════════════════════════════════════════════════════

═══════════════════════════════════════════════════════════
DYNAMIC VISUAL ADAPTATION
═══════════════════════════════════════════════════════════
For each entity type listed in key_entities, dynamically generate a sprite using Phaser graphics:
- The shape, color, and pattern must reflect the entity's concept.
- Example:
  - Alien → glowing green oval with eyes and tentacles and legs and hands.
  - Lion → orange/yellow shape with a mane (spikes or layered circles) and legs and hands.
  - Robot → metallic gray box with antenna lights and legs and hands.
  - Ghost → floating blob with trailing bottom and hands.
- Background must match setting (space, jungle, ruins, neon city, etc.)
- No static rectangles or circles — combine multiple shapes and effects.
- Use procedural drawing: gradients, polygons, arcs, layered fills, and particle-like sparkles.
- Animate some parts slightly (rotation, pulsing, fading).

function generateSprite(scene, key, type) {{
  const g = scene.make.graphics({{ x: 0, y: 0, add: false }});
  
  if (type === "alien") {{
    // Alien: glowing green oval with tentacles
    g.fillStyle(0x00ff88);
    g.fillEllipse(16, 16, 14, 20);
    for (let i = 0; i < 5; i++) {{
      const x = 16 + Math.cos(i * 1.3) * 12;
      g.fillStyle(0x00dd77);
      g.fillRect(x - 2, 28, 4, 6);
    }}
    g.lineStyle(2, 0xffffff, 0.6);
    g.strokeEllipse(16, 16, 14, 20);
  }}
  else if (type === "lion") {{
    // Lion: mane of spikes + face center
    for (let i = 0; i < 16; i++) {{
      const angle = i * (Math.PI / 8);
      g.fillStyle(0xffaa00);
      g.fillTriangle(16, 16,
                     16 + Math.cos(angle) * 20,
                     16 + Math.sin(angle) * 20,
                     16 + Math.cos(angle + 0.3) * 10,
                     16 + Math.sin(angle + 0.3) * 10);
    }}
    g.fillStyle(0xffcc33);
    g.fillCircle(16, 16, 10);
  }}
  else if (type === "robot") {{
    // Robot: metallic body with lights
    g.fillStyle(0x888888);
    g.fillRect(4, 4, 24, 24);
    g.fillStyle(0x00ffff);
    g.fillRect(10, 10, 4, 4);
    g.fillRect(18, 10, 4, 4);
    g.fillStyle(0xff0000);
    g.fillRect(12, 20, 8, 3);
  }}
  else if (type === "ghost") {{
    g.fillStyle(0x99ffff);
    g.fillEllipse(16, 16, 14, 18);
    g.fillStyle(0x77ddff);
    g.fillRect(8, 24, 16, 4);
  }}
  else {{
    // Default (arcade orb)
    g.fillStyle(0xff00ff);
    g.fillCircle(16, 16, 14);
    g.lineStyle(3, 0xffffff, 0.7);
    g.strokeCircle(16, 16, 14);
  }}

  g.generateTexture(key, 32, 32);
  g.destroy();
}}


// ═══════════════════════════════════════════════════════════
// CREATE: Game Initialization
// ═══════════════════════════════════════════════════════════
function create() {{
    gameScene = this;  // DO NOT CHANGE: Store scene reference
    
    // Player setup
    player = this.physics.add.sprite(400, 300, 'player');
    player.setCollideWorldBounds(true);
    
    // Groups
    entities = this.physics.add.group();
    projectiles = this.physics.add.group();
    
    // ADAPT INPUT: Match these to your control specification
    keys = this.input.keyboard.addKeys({{
        W: 'W',
        A: 'A',
        S: 'S',
        D: 'D',
        SPACE: 'SPACE',
        UP: 'UP',
        DOWN: 'DOWN',
        LEFT: 'LEFT',
        RIGHT: 'RIGHT'
    }});
    
    // Mouse controls (if needed)
    this.input.on('pointerdown', handlePointerDown);
    
    // Collisions
    this.physics.add.overlap(player, entities, handlePlayerEntityCollision, null, this);
    this.physics.add.overlap(projectiles, entities, handleProjectileEntityCollision, null, this);
    
    // UI
    scoreText = this.add.text(16, 16, 'Score: 0', {{ fontSize: '24px', fill: '#fff' }});
    healthText = this.add.text(16, 48, 'Health: 100', {{ fontSize: '20px', fill: '#0f0' }});
    levelText = this.add.text(16, 76, 'Level: 1', {{ fontSize: '20px', fill: '#ff0' }});
    
    // ADAPT: Match instruction text to actual controls
    instructionsText = this.add.text(550, 16, 
        'WASD: Move\\nMouse: Shoot', 
        {{ fontSize: '16px', fill: '#aaa', align: 'right' }}
    );
    
    // ADAPT: Spawn initial entities appropriate for your game
    for (let i = 0; i < 5; i++) {{
        spawnEntity();
    }}
    
    // ADAPT: Entity spawning timer
    this.time.addEvent({{
        delay: 2000,
        callback: spawnEntity,
        loop: true
    }});
}}

// ═══════════════════════════════════════════════════════════
// UPDATE: Main Game Loop
// ═══════════════════════════════════════════════════════════
function update(time, delta) {{
    if (gameOver) return;
    
    // ADAPT MOVEMENT: Choose based on genre
    
    // Option 1: Top-down (shooters, top-down games)
    player.setVelocity(0);
    if (keys.W.isDown || keys.UP.isDown) player.setVelocityY(-200);
    if (keys.S.isDown || keys.DOWN.isDown) player.setVelocityY(200);
    if (keys.A.isDown || keys.LEFT.isDown) player.setVelocityX(-200);
    if (keys.D.isDown || keys.RIGHT.isDown) player.setVelocityX(200);
    
    // Option 2: Platformer (uncomment if platformer)
    // player.setVelocityX(0);
    // if (keys.A.isDown || keys.LEFT.isDown) player.setVelocityX(-160);
    // if (keys.D.isDown || keys.RIGHT.isDown) player.setVelocityX(160);
    // if (keys.SPACE.isDown && player.body.touching.down) player.setVelocityY(-330);
    
    // Option 3: Auto-runner (uncomment if runner)
    // player.setVelocityX(200);
    // if (keys.SPACE.isDown && player.body.touching.down) player.setVelocityY(-400);
    
    // ADAPT AI: Entity behavior every frame
    entities.getChildren().forEach(function(entity) {{
        // Chase AI (for enemies)
        gameScene.physics.moveToObject(entity, player, 60);
        
        // OR Patrol AI (uncomment for platformer):
        // if (!entity.getData('dir')) entity.setData('dir', 1);
        // entity.setVelocityX(50 * entity.getData('dir'));
        // if (entity.x < 50 || entity.x > 750) entity.setData('dir', -entity.getData('dir'));
    }});
    
    // ADAPT ACTION: Shooting or other action
    if (keys.SPACE.isDown && time > (player.lastFired || 0) + 300) {{
        fireProjectile();
        player.lastFired = time;
    }}
    
    // Cleanup off-screen projectiles
    projectiles.getChildren().forEach(function(proj) {{
        if (proj.x < -50 || proj.x > 850 || proj.y < -50 || proj.y > 650) proj.destroy();
    }});
}}

// ═══════════════════════════════════════════════════════════
// INPUT HANDLERS
// ═══════════════════════════════════════════════════════════
function handlePointerDown(pointer) {{
    if (gameOver) return;
    fireProjectile(pointer.x, pointer.y);
}}

// ═══════════════════════════════════════════════════════════
// GAME MECHANICS
// ═══════════════════════════════════════════════════════════
function spawnEntity() {{
    // ADAPT: Spawn position based on game type
    var x = Phaser.Math.Between(50, 750);
    var y = Phaser.Math.Between(50, 150);
    
    var entity = entities.create(x, y, 'entity');
}}

function fireProjectile(targetX, targetY) {{
    var projectile = projectiles.create(player.x, player.y, 'projectile');
    
    if (targetX !== undefined) {{
        // Mouse aiming
        var angle = Phaser.Math.Angle.Between(player.x, player.y, targetX, targetY);
        gameScene.physics.velocityFromRotation(angle, 400, projectile.body.velocity);
    }} else {{
        // Fixed direction
        projectile.setVelocityY(-400);
    }}
}}

// ═══════════════════════════════════════════════════════════
// COLLISION HANDLERS
// ═══════════════════════════════════════════════════════════
function handlePlayerEntityCollision(player, entity) {{
    // ADAPT: Define what happens when player touches entity
    health -= 10;
    healthText.setText('Health: ' + health);
    healthText.setColor(health > 50 ? '#0f0' : '#f00');
    entity.destroy();
    
    if (health <= 0) endGame('Health depleted!');
}}

function handleProjectileEntityCollision(projectile, entity) {{
    projectile.destroy();
    entity.destroy();
    score += 10;
    scoreText.setText('Score: ' + score);
    
    // Level progression
    if (entities.countActive(true) === 0) {{
        level++;
        levelText.setText('Level: ' + level);
        for (let i = 0; i < level * 3; i++) spawnEntity();
    }}
}}

// ═══════════════════════════════════════════════════════════
// GAME STATE
// ═══════════════════════════════════════════════════════════
function endGame(reason) {{
    if (gameOver) return;
    gameOver = true;
    gameScene.physics.pause();
    player.setTint(0xff0000);
    
    gameScene.add.text(400, 250, 'GAME OVER', {{
        fontSize: '64px', fill: '#fff'
    }}).setOrigin(0.5);
    
    gameScene.add.text(400, 320, reason, {{
        fontSize: '24px', fill: '#aaa'
    }}).setOrigin(0.5);
    
    gameScene.add.text(400, 360, 'Final Score: ' + score, {{
        fontSize: '32px', fill: '#ff0'
    }}).setOrigin(0.5);
    
    gameScene.add.text(400, 420, 'Click to Restart', {{
        fontSize: '20px', fill: '#888'
    }}).setOrigin(0.5);
    
    gameScene.input.once('pointerdown', function() {{
        gameScene.scene.restart();
        score = 0; health = 100; level = 1; gameOver = false;
    }});
}}

</script>
</body>
</html>

═══════════════════════════════════════════════════════════
ADAPTATION INSTRUCTIONS FOR YOUR GAME
═══════════════════════════════════════════════════════════

Genre: {genre}
Required Controls: {', '.join(controls) if isinstance(controls, list) else controls}

ADAPT THESE SECTIONS (marked with "ADAPT:" comments):

1. GRAVITY SETTING:
   - Top-down shooter/puzzle: gravity.y = 0
   - Platformer/runner: gravity.y = 300
   - Your game: Choose based on genre

2. MOVEMENT PATTERN:
   - Use Option 1 (top-down) for: shooter, puzzle, top-down
   - Use Option 2 (platformer) for: platformer, side-scroller
   - Use Option 3 (auto-runner) for: endless runner, auto-scroller

3. ENTITY AI:
   - Use chase AI for: enemies in shooters
   - Use patrol AI for: enemies in platformers
   - Use falling/static for: obstacles in runners/puzzles

4. INSTRUCTION TEXT:
   - MUST match actual controls in code
   - If using WASD, text must say "WASD: Move"
   - If using arrow keys only, text must say "Arrow Keys: Move"

5. SPAWN LOGIC:
   - Adjust spawn positions for your game type
   - Adjust spawn rate based on difficulty

6. COLLISION BEHAVIOR:
   - Define what happens when player touches entities
   - Define what happens when projectiles hit entities
   - Match game rules from design document

═══════════════════════════════════════════════════════════
CRITICAL RULES - DO NOT BREAK THESE
═══════════════════════════════════════════════════════════

✅ KEEP UNCHANGED:
- gameScene = this pattern
- graphics.generateTexture() sprite creation
- keys defined ONCE in create()
- physics.velocityFromRotation() usage
- Collision callback structure with proper context

❌ NEVER DO:
- Use this.load.image() with base64 or URLs
- Define keys inside update() function
- Use 'this' in callbacks without storing gameScene
- Call moveToObject() only once at spawn
- Display UI text that doesn't match controls
- Use player.rotation without setting it

═══════════════════════════════════════════════════════════
FINAL OUTPUT
═══════════════════════════════════════════════════════════

Return ONLY the complete HTML code above with adaptations made.
Start with <!DOCTYPE html>
No markdown fences. No explanations. Just the working HTML.
"""
    
    state["game_prompt"] = prompt
    
    log_timestamp(f"✅ Enhanced template prompt built ({len(prompt)} chars)")
    
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
    print("---NODE: REVIEW GAME CODE---")
    print("="*60)
    
    code = state.get("generated_code", "")
    engine = state.get("engine_choice", "HTML_CANVAS")
    fix_iteration = state.get("fix_iteration", 0)
    
    log_timestamp(f"🔍 Starting code review (iteration #{fix_iteration})...")
    log_timestamp(f"📊 Code stats: {len(code)} chars, {code.count('function')} functions")
    
    # ✅ Log code preview
    print("\n" + "-"*60)
    print("📄 CODE BEING REVIEWED (first 1500 chars):")
    print("-"*60)
    print(code[:1500])
    if len(code) > 1500:
        print(f"\n... [{len(code) - 1500} more chars] ...")
    print("-"*60 + "\n")
    
    # ✅ Log code structure
    log_timestamp("🏗️  Code structure analysis:")
    log_timestamp(f"   - Contains <!DOCTYPE>: {code.strip().startswith('<!DOCTYPE')}")
    log_timestamp(f"   - Contains <html>: {('<html' in code.lower())}")
    log_timestamp(f"   - Contains <canvas>: {('nvas' in code.lower())}")
    log_timestamp(f"   - Containsins <script>: {('<script' in code.lower())}")
    log_timestamp(f"   - Script tags count: {code.lower().count('<script')}")
    log_timestamp(f"   - Function definitions: {code.count('function ')}")
    log_timestamp(f"   - Event listeners: {code.count('addEventListener')}")
    log_timestamp(f"   - Game loop patterns: {code.count('requestAnimationFrame') + code.count('setInterval')}")
    
    # Generic principle-based review prompt
    prompt = f"""
You are a game QA tester reviewing a {engine} browser game for playability.

Review the code using these PRINCIPLES:

1. CONSISTENCY PRINCIPLE
   Does the user interface match the implementation?
2. COMPLETENESS PRINCIPLE
   Are core mechanics fully implemented?
3. VISIBILITY PRINCIPLE
   Can the player actually see and interact with the game?
4. FUNCTIONAL PRINCIPLE
   Does the core gameplay loop work?

Review for PLAYABILITY, not code quality. Ask yourself:
"If I loaded this in a browser, could I play and understand what's happening?"

Mark as FAIL if:
- Core mechanic doesn't work as intended
- UI text contradicts actual controls
- Progression/state variables display but never change
- Game provides no feedback on critical events
- Sprites invisible or game entities don't spawn

what i have gave you are just examples of how to review , think and analyze the code for more potentiol errors , it does not needs to be perfect but atleast should not crash.

Respond with JSON only (no markdown):
{{
  "status": "pass" | "fail",
  "issues": ["concise issue description with what's wrong and why it breaks gameplay"],
  "suggestions": "constructive feedback"
}}

Code to review (first 8000 chars):
{code}
"""
    
    log_timestamp("⏳ Sending code to reviewer LLM...")
    out = llm_invoke_text(prompt)
    
    # ✅ Log raw review response
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
            log_timestamp(f"❌ Issues found ({len(issues)}):")
            for i, issue in enumerate(issues, 1):
                log_timestamp(f"   {i}. {issue}")
        else:
            log_timestamp("✅ Code passed review!")
        
        if review.get("suggestions"):
            log_timestamp(f"💡 Suggestions: {review['suggestions'][:200]}")
            
    except Exception as e:
        log_timestamp(f"⚠️  Review parsing failed: {str(e)}")
        log_timestamp("⚠️  Raw response preview: " + out[:300])
        review = {
            "status": "fail",
            "issues": ["Could not parse reviewer output.", "Review response was malformed."],
            "suggestions": out[:1000]
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
    engine = state.get("engine_choice", "HTML_CANVAS")
    
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
    
    log_timestamp(f"🛠️ Applying user feedback: {feedback[:120]}...")
    
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
    
    out = llm_invoke_text(prompt)
    
    fixed = re.sub(r"^```(?:html)?\s*", "", out.strip())
    fixed = re.sub(r"\s*```$", "", fixed)
    if fixed.strip() == code.strip():
        log_timestamp("⚠️ No visible change detected. Reinforcing feedback...")
        stronger_prompt = f"""
    Forcefully apply the feedback below and ensure visible or functional difference.
    Feedback: {feedback}
    Code: {code}
    """
    out = llm_invoke_text(stronger_prompt)
    fixed = re.sub(r"^```(?:html)?\s*", "", out.strip())
    fixed = re.sub(r"\s*```$", "", fixed)

    state["final_code"] = fixed
    log_timestamp(f"✅ Feedback applied, new code length: {len(fixed)} chars")
    print(f"xxxxxx==========xxxxxx {fixed} xxxxxx==========xxxxxx")
    
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
workflow.add_node("design_game_blueprint", design_game_blueprint)
workflow.add_node("engine_decision", engine_decision)
workflow.add_node("build_code_prompt", build_code_prompt)
workflow.add_node("generate_game_code", generate_game_code)
workflow.add_node("review_code", review_code)
workflow.add_node("fix_game_code", fix_game_code)
workflow.add_node("finalize_output", finalize_output)
workflow.add_node("collect_user_feedback", collect_user_feedback)
workflow.add_node("apply_feedback_to_code", apply_feedback_to_code)
workflow.add_node("verify_feedback_applied", verify_feedback_applied)

# Entry point
workflow.set_entry_point("intent_analysis")

# Edges (linear flow + conditional review->fix loop)
workflow.add_edge("intent_analysis", "collect_user_idea")
workflow.add_edge("collect_user_idea", "generate_questions")
workflow.add_edge("generate_questions", "collect_user_answers")
workflow.add_edge("collect_user_answers", "validate_inputs")
workflow.add_edge("validate_inputs", "design_game_blueprint")
workflow.add_edge("design_game_blueprint", "engine_decision")
workflow.add_edge("engine_decision", "build_code_prompt")
workflow.add_edge("build_code_prompt", "generate_game_code")
workflow.add_edge("generate_game_code", "review_code")
workflow.add_edge("collect_user_feedback", "apply_feedback_to_code")
workflow.add_edge("apply_feedback_to_code", "verify_feedback_applied")
workflow.add_edge("verify_feedback_applied", "review_code")

# workflow.add_edge("apply_feedback_to_code", "review_code")

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

if __name__ == "__main__":
    print("="*60)
    print("GameForge LangGraph agent compiled as `game_agent_app`.")
    print("="*60)
