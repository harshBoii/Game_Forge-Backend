from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Allow CORS (so Next.js frontend can access it)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/api/generate")
async def generate_game(req: Request):
    data = await req.json()
    prompt = data.get("prompt", "")

    system_prompt = """
    You are a JavaScript game generator.
    Given a user prompt, return playable 2D HTML + CSS + JavaScript code that
    runs instantly in the browser.
    Output a complete <html> document with inline <style> and <script>.
    Avoid external dependencies.
    """

    # ✅ New syntax for OpenAI v1+
    completion = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    html_code = completion.choices[0].message.content
    return {"html": html_code}
