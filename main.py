# # main.py
# import os
# import uuid
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict

# from langgraph.types import Command  # ✅ Import Command
# from agent import game_agent_app, GameAgentState

# # -------------------------------
# # ✅ FastAPI + CORS Setup
# # -------------------------------
# app = FastAPI(title="GameForge AI Backend", version="2.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------------------
# # ✅ Request Models
# # -------------------------------
# class StartRequest(BaseModel):
#     """Model for starting a new generation session"""
#     prompt: str

# class ResumeRequest(BaseModel):
#     """Model for resuming after an interrupt"""
#     session_id: str
#     answers: List[Dict[str, str]]

# class FeedbackRequest(BaseModel):
#     """Model for resuming after feedback interrupt"""
#     session_id: str
#     feedback: str

# # -------------------------------
# # ✅ Start New Game Session
# # -------------------------------
# @app.post("/api/start")
# async def start_game(req: StartRequest):
#     """
#     Starts a new game generation session.
#     The agent runs until it hits the first interrupt (questions).
#     """
#     try:
#         # ✅ Generate session ID upfront
#         session_id = str(uuid.uuid4())
        
#         # Initialize new state
#         state: GameAgentState = {
#             "session_id": session_id,
#             "user_raw_input": req.prompt.strip(),
#         }

#         # ✅ Run the LangGraph agent with proper config
#         config = {"configurable": {"thread_id": session_id}}
#         result = await game_agent_app.ainvoke(state, config=config)

#         # ✅ Check for interrupt
#         if "__interrupt__" in result:
#             interrupts = result["__interrupt__"]
#             if interrupts:
#                 # Get the first interrupt payload
#                 interrupt_data = interrupts[0].value if hasattr(interrupts[0], 'value') else interrupts[0]
#                 return {
#                     "type": "interrupt",
#                     "session_id": session_id,
#                     "message": interrupt_data.get("message"),
#                     "questions": interrupt_data.get("questions", [])
#                 }

#         # Otherwise, return final output
#         return {
#             "type": "success",
#             "session_id": session_id,
#             "engine_choice": result.get("engine_choice"),
#             "reasoning": result.get("engine_reasoning"),
#             "summary": result.get("final_summary"),
#             "html": result.get("final_response", {}).get("html", "")
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------
# # ✅ Resume Existing Session
# # -------------------------------
# @app.post("/api/resume")
# async def resume_game(req: ResumeRequest):
#     """
#     Resumes an interrupted session.
#     The user has answered the AI-generated questions,
#     and we continue the flow from where it paused.
#     """
#     try:
#         # ✅ Resume with Command(resume=answers)
#         config = {"configurable": {"thread_id": req.session_id}}
#         result = await game_agent_app.ainvoke(
#             Command(resume=req.answers),  # ✅ Pass answers as resume value
#             config=config
#         )

#         # Check if it interrupts again
#         if "__interrupt__" in result:
#             interrupts = result["__interrupt__"]
#             if interrupts:
#                 interrupt_data = interrupts[0].value if hasattr(interrupts[0], 'value') else interrupts[0]
#                 return {
#                     "type": "interrupt",
#                     "session_id": req.session_id,
#                     "message": interrupt_data.get("message"),
#                     "questions": interrupt_data.get("questions", [])
#                 }

#         # Otherwise, we're done
#         return {
#             "type": "success",
#             "session_id": req.session_id,
#             "engine_choice": result.get("engine_choice"),
#             "reasoning": result.get("engine_reasoning"),
#             "summary": result.get("final_summary"),
#             "html": result.get("final_response", {}).get("html", "")
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/api/feedback")
# async def resume_after_feedback(req: FeedbackRequest):
#     """
#     Continues the session after user gives feedback on generated game.
#     The feedback string is sent back into the LangGraph workflow,
#     which then applies the feedback and regenerates code.
#     """
#     try:
#         config = {"configurable": {"thread_id": req.session_id}}

#         # Resume from the feedback interrupt node
#         result = await game_agent_app.ainvoke(
#             Command(resume=req.feedback),  # ✅ Pass the feedback text
#             config=config
#         )

#         # Handle possible re-interrupt (for another feedback round)
#         if "__interrupt__" in result:
#             interrupts = result["__interrupt__"]
#             if interrupts:
#                 interrupt_data = interrupts[0].value if hasattr(interrupts[0], 'value') else interrupts[0]
#                 return {
#                     "type": "interrupt",
#                     "session_id": req.session_id,
#                     "message": interrupt_data.get("message"),
#                 }

#         # ✅ Otherwise return updated result
#         return {
#             "type": "success",
#             "session_id": req.session_id,
#             "summary": result.get("final_summary"),
#             "html": result.get("final_response", {}).get("html", ""),
#             "feedback_iteration": result.get("feedback_iteration", 0)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # -------------------------------
# # ✅ Health Check
# # -------------------------------
# @app.get("/")
# async def root():
#     return {"status": "ok", "message": "GameForge AI backend is live 🚀"}

# # -------------------------------
# # ✅ Run Locally
# # -------------------------------
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
# main.py
import os
import uuid
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict

from langgraph.types import Command
from agent import (
    game_agent_app,
    GameAgentState,
    apply_feedback_to_code,
    review_code,
    fix_game_code,
    finalize_output,
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

# ============================================================
# ✅ Request Models
# ============================================================
class StartRequest(BaseModel):
    prompt: str


class ResumeRequest(BaseModel):
    session_id: str
    answers: List[Dict[str, str]]


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
        state: GameAgentState = {
            "session_id": session_id,
            "user_raw_input": req.prompt.strip(),
        }

        config = {"configurable": {"thread_id": session_id}}
        result = await game_agent_app.ainvoke(state, config=config)

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
    Does NOT rely on LangGraph resume (since the graph finished).
    Instead, it loads the last generated HTML, applies changes via
    apply_feedback_to_code → review_code → fix_game_code → finalize_output.
    """
    sid = req.session_id
    if sid not in FINAL_HTML_STORE:
        raise HTTPException(status_code=404, detail="No saved game found for this session ID.")

    # Build new state
    state: GameAgentState = {
        "session_id": sid,
        "final_code": FINAL_HTML_STORE[sid],
        "generated_code": FINAL_HTML_STORE[sid],
        "user_feedback": req.feedback,
        "feedback_iteration": 1,
        "fix_iteration": 0,
        "feedback_history": [{"iteration": 1, "feedback": req.feedback}],
    }

    try:
        # 1️⃣ Apply feedback
        state = apply_feedback_to_code(state)

        # 2️⃣ Review new code
        state = review_code(state)

        # 3️⃣ If failed, attempt auto-fix up to 3 times
        max_fix_rounds = 3
        while (
            state.get("review_notes", {}).get("status") == "fail"
            and state.get("fix_iteration", 0) < max_fix_rounds
        ):
            state = fix_game_code(state)
            state = review_code(state)

        # 4️⃣ Finalize and persist updated HTML
        state = finalize_output(state)
        new_html = state.get("final_response", {}).get("html", "")
        if new_html:
            FINAL_HTML_STORE[sid] = new_html

        return {
            "type": "success",
            "session_id": sid,
            "summary": state.get("final_summary"),
            "html": new_html,
            "feedback_iteration": state.get("feedback_iteration", 1),
            "review_notes": state.get("review_notes", {}),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# ✅ Health Check Endpoint
# ============================================================
@app.get("/")
async def root():
    return {"status": "ok", "message": "GameForge AI backend is running"}


# ============================================================
# ✅ Local Run
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,
    )
