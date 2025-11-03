
# main.py
import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from langgraph.types import Command
from agent__ import (
    game_agent_app,
    GameAgentState,
    apply_feedback_to_code,
    review_code,
    fix_game_code,
    finalize_output,
    get_llm
)

# ============================================================
# ✅ FastAPI Setup
# ============================================================
app = FastAPI(title="GameForge AI Backend", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ✅ In-Memory Store (replace with Prisma/Neon in production)
# ============================================================
FINAL_HTML_STORE: Dict[str, str] = {}
GAME_SESSIONS: Dict[str, GameAgentState] = {}

# ============================================================
# ✅ Request Models
# ============================================================
class StartRequest(BaseModel):
    prompt: str
    model: str | None = None

class ResumeRequest(BaseModel):
    session_id: str
    answers: List[Dict[str, str]]
    model: str | None = None

class FeedbackRequest(BaseModel):
    session_id: str
    feedback: str


# ============================================================
# ✅ Start a New Generation Session
# ============================================================
@app.post("/api/start")
async def start_game(req: StartRequest):
    """
    Starts a new game generation session.
    Runs until it reaches the first interrupt (question set).
    """
    try:
        session_id = str(uuid.uuid4())
        model_name = req.model or os.getenv("ANTHROPIC_MODEL")
        
        # ✅ dynamically create LLM
        llm_instance = get_llm(model_name)

        # ✅ inject LLM into your agent pipeline
        game_agent_app.llm = llm_instance

        state: GameAgentState = {
            "session_id": session_id,
            "user_raw_input": req.prompt.strip(),
            "model_name": model_name

        }

        config = {"configurable": {"thread_id": session_id}}
        result = await game_agent_app.ainvoke(state, config=config)
        
        GAME_SESSIONS[session_id] = result

        # Handle interrupt (asking user questions)
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value
            return {
                "type": "interrupt",
                "session_id": session_id,
                "message": interrupt_data.get("message"),
                "questions": interrupt_data.get("questions", []),
            }

        # Normal completion
        html = result.get("final_response", {}).get("html", "")
        if html:
            FINAL_HTML_STORE[session_id] = html

        return {
            "type": "success",
            "session_id": session_id,
            "engine_choice": result.get("engine_choice"),
            "reasoning": result.get("engine_reasoning"),
            "summary": result.get("final_summary"),
            "html": html,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ✅ Resume Existing Session (after questions answered)
# ============================================================
@app.post("/api/resume")
async def resume_game(req: ResumeRequest):
    """
    Resumes a generation session after user answered design questions.
    Continues until the game code is produced.
    """
    try:
        config = {"configurable": {"thread_id": req.session_id}}
        result = await game_agent_app.ainvoke(Command(resume=req.answers), config=config)

        # Handle another interrupt (if more questions)
        if "__interrupt__" in result:
            interrupt_data = result["__interrupt__"][0].value
            return {
                "type": "interrupt",
                "session_id": req.session_id,
                "message": interrupt_data.get("message"),
                "questions": interrupt_data.get("questions", []),
            }

        # Save the generated HTML for future feedback iterations
        html = result.get("final_response", {}).get("html", "")
        if html:
            FINAL_HTML_STORE[req.session_id] = html

        GAME_SESSIONS[req.session_id] = result

        return {
            "type": "success",
            "session_id": req.session_id,
            "engine_choice": result.get("engine_choice"),
            "reasoning": result.get("engine_reasoning"),
            "summary": result.get("final_summary"),
            "html": html,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ✅ Apply User Feedback and Regenerate Game
# ============================================================
@app.post("/api/feedback")
async def feedback_endpoint(req: FeedbackRequest):
    """
    Applies user feedback to an already generated game.
    """
    sid = req.session_id
    
    # ✅ Enhanced logging - Entry point
    print("\n" + "="*70)
    print("🔄 FEEDBACK ENDPOINT CALLED")
    print("="*70)
    print(f"📋 Session ID: {sid}")
    print(f"💬 Feedback: {req.feedback}")
    print(f"📊 Total sessions in memory: {len(GAME_SESSIONS)}")
    print(f"🔑 Available session IDs: {list(GAME_SESSIONS.keys())}")
    
    if sid not in GAME_SESSIONS:
        error_msg = f"Session ID '{sid}' not found in GAME_SESSIONS"
        print(f"\n❌ ERROR: {error_msg}")
        print(f"Available sessions: {list(GAME_SESSIONS.keys())}")
        raise HTTPException(
            status_code=404, 
            detail={
                "error": error_msg,
                "available_sessions": list(GAME_SESSIONS.keys()),
                "total_sessions": len(GAME_SESSIONS)
            }
        )

    try:
        # ✅ Load the complete previous state
        print("\n📦 Loading previous state...")
        previous_state = GAME_SESSIONS[sid]
        print(f"✅ Previous state loaded")
        print(f"   Keys in state: {list(previous_state.keys())}")
        print(f"   Engine choice: {previous_state.get('engine_choice', 'NOT_SET')}")
        print(f"   Has generated_code: {'generated_code' in previous_state}")
        print(f"   Has final_code: {'final_code' in previous_state}")
        print(f"   Current feedback iteration: {previous_state.get('feedback_iteration', 0)}")
        
        # ✅ Validate required fields
        code_field = None
        if "generated_code" in previous_state and previous_state["generated_code"]:
            code_field = "generated_code"
        elif "final_code" in previous_state and previous_state["final_code"]:
            code_field = "final_code"
            previous_state["generated_code"] = previous_state["final_code"]
        elif "final_response" in previous_state and isinstance(previous_state["final_response"], dict):
            if "html" in previous_state["final_response"]:
                code_field = "final_response.html"
                previous_state["generated_code"] = previous_state["final_response"]["html"]
        
        if not code_field:
            error_msg = "No game code found in session state"
            print(f"\n❌ ERROR: {error_msg}")
            print(f"   Checked fields: generated_code, final_code, final_response.html")
            print(f"   Available state keys: {list(previous_state.keys())}")
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": error_msg,
                    "available_keys": list(previous_state.keys()),
                    "hint": "Game may not have been fully generated. Try /api/resume first."
                }
            )
        
        print(f"✅ Game code found in: {code_field}")
        print(f"   Code length: {len(previous_state['generated_code'])} characters")
        
        # ✅ Build updated state
        print("\n🔧 Building updated state...")
        state: GameAgentState = {
            **previous_state,
            "user_feedback": req.feedback,
            "feedback_iteration": previous_state.get("feedback_iteration", 0) + 1,
            "fix_iteration": 0,
            "feedback_history": previous_state.get("feedback_history", []) + [
                {"iteration": previous_state.get("feedback_iteration", 0) + 1, "feedback": req.feedback}
            ],
        }
        print(f"✅ State updated")
        print(f"   New feedback iteration: {state['feedback_iteration']}")

        # 1️⃣ Apply feedback
        print("\n" + "-"*70)
        print("1️⃣  STEP: Applying feedback to code...")
        print("-"*70)
        try:
            state = apply_feedback_to_code(state)
            print(f"✅ Feedback applied successfully")
            print(f"   Updated code length: {len(state.get('generated_code', ''))} characters")
        except Exception as e:
            print(f"\n❌ ERROR in apply_feedback_to_code:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            raise

        # 2️⃣ Review new code
        print("\n" + "-"*70)
        print("2️⃣  STEP: Reviewing updated code...")
        print("-"*70)
        try:
            state = review_code(state)
            review_status = state.get("review_notes", {}).get("status", "unknown")
            print(f"✅ Code review completed")
            print(f"   Review status: {review_status}")
            if review_status == "fail":
                issues = state.get("review_notes", {}).get("issues", [])
                print(f"   ⚠️  Issues found: {len(issues)}")
                for i, issue in enumerate(issues[:3], 1):
                    print(f"      {i}. {issue}")
        except Exception as e:
            print(f"\n❌ ERROR in review_code:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            raise

        # 3️⃣ Fix loop if needed
        max_fix_rounds = 3
        fix_count = 0
        print("\n" + "-"*70)
        print("3️⃣  STEP: Auto-fix loop (if needed)...")
        print("-"*70)
        
        while (
            state.get("review_notes", {}).get("status") == "fail"
            and state.get("fix_iteration", 0) < max_fix_rounds
        ):
            fix_count += 1
            print(f"\n🔧 Fix attempt {fix_count}/{max_fix_rounds}...")
            try:
                state = fix_game_code(state)
                state = review_code(state)
                new_status = state.get("review_notes", {}).get("status")
                print(f"   Review status after fix: {new_status}")
            except Exception as e:
                print(f"   ❌ ERROR during fix attempt {fix_count}:")
                print(f"      Error type: {type(e).__name__}")
                print(f"      Error message: {str(e)}")
                raise
        
        if fix_count == 0:
            print("✅ No fixes needed - code passed review")

        # 4️⃣ Finalize
        print("\n" + "-"*70)
        print("4️⃣  STEP: Finalizing output...")
        print("-"*70)
        try:
            state = finalize_output(state)
            print(f"✅ Output finalized")
        except Exception as e:
            print(f"\n❌ ERROR in finalize_output:")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Error message: {str(e)}")
            raise
        
        # ✅ Update stored state
        GAME_SESSIONS[sid] = state
        new_html = state.get("final_response", {}).get("html", "")
        
        print("\n" + "="*70)
        print("✅ FEEDBACK PROCESSING COMPLETED SUCCESSFULLY")
        print("="*70)

        return {
            "type": "success",
            "session_id": sid,
            "summary": state.get("final_summary"),
            "html": new_html,
            "feedback_iteration": state.get("feedback_iteration", 1),
            "review_notes": state.get("review_notes", {}),
        }

    except HTTPException:
        raise
        
    except Exception as e:
        import traceback
        import sys
        
        print("\n" + "="*70)
        print("❌ FATAL ERROR IN FEEDBACK ENDPOINT")
        print("="*70)
        print(f"\n🔴 Error Type: {type(e).__name__}")
        print(f"🔴 Error Message: {str(e)}")
        print(f"\n📍 Full Stack Trace:")
        print("-"*70)
        traceback.print_exc(file=sys.stdout)
        print("-"*70)
        
        if 'state' in locals():
            print(f"\n📊 State at failure:")
            print(f"   Session ID: {state.get('session_id', 'N/A')}")
            print(f"   Engine choice: {state.get('engine_choice', 'N/A')}")
            print(f"   State keys: {list(state.keys())}")
        
        print(f"\n💬 Feedback: {req.feedback[:300]}")
        print("="*70)
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "feedback_preview": req.feedback[:200],
                "session_id": sid,
                "hint": "Check server console logs for detailed stack trace"
            }
        )
# ============================================================
# ✅ Health Check Endpoint
# ============================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "GameForge AI backend is running"}

