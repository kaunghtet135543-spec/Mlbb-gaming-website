import os
import time
from collections import defaultdict, deque

import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024
allowed_origins = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "").split(",") if x.strip()]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 12
request_times = defaultdict(deque)

@app.get("/health")
def health():
    return jsonify(status="ok", api_key_configured=bool(GEMINI_API_KEY))

@app.post("/api/chat")
def chat():
    ip = request.remote_addr or "unknown"
    now = time.monotonic()
    recent = request_times[ip]
    while recent and now - recent[0] > WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= MAX_REQUESTS_PER_WINDOW:
        return jsonify(error="ခဏစောင့်ပြီးမှ ပြန်မေးပါ။"), 429
    recent.append(now)

    if not GEMINI_API_KEY:
        return jsonify(error="Backend မှာ GEMINI_API_KEY မသတ်မှတ်ရသေးပါ။"), 503
    body = request.get_json(silent=True) or {}
    message = body.get("message")
    if not isinstance(message, str) or not message.strip() or len(message) > 1200:
        return jsonify(error="မေးခွန်းကို စာလုံး ၁၂၀၀ အတွင်း ထည့်ပါ။"), 400

    history = body.get("history", [])
    if not isinstance(history, list):
        history = []
    lines = []
    for item in history[-8:]:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            lines.append(("အသုံးပြုသူ" if role == "user" else "AI") + ": " + content[:1200])
    lines.append("အသုံးပြုသူ: " + message.strip())
    payload = {
        "model": GEMINI_MODEL,
        "store": False,
        "system_instruction": (
            "You are the AI assistant inside MLBB Gaming Zone. Reply in the language the user uses, "
            "including Burmese when they write Burmese. Help with Mobile Legends heroes and gameplay. "
            "Do not claim community build advice is an official recommendation. If unsure about current "
            "patch details, say that it may have changed."
        ),
        "input": "\n".join(lines),
    }
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
    except requests.RequestException:
        return jsonify(error="Gemini API သို့ ချိတ်ဆက်မရပါ။ ခဏကြာပြီး ပြန်စမ်းပါ။"), 502
    if response.status_code == 429:
        return jsonify(error="AI API အသုံးပြုမှုကန့်သတ်ချက် ပြည့်နေပါတယ်။ နောက်မှပြန်စမ်းပါ။"), 429
    if not response.ok:
        app.logger.warning("Gemini returned HTTP %s", response.status_code)
        return jsonify(error="Gemini API မှ အမှားပြန်လာပါတယ်။ API key/model setting ကို စစ်ပါ။"), 502
    try:
        data = response.json()
        texts = [
            part["text"]
            for step in data.get("steps", [])
            if step.get("type") == "model_output"
            for part in step.get("content", [])
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
        reply = "\n".join(texts).strip()
    except (ValueError, TypeError, AttributeError):
        reply = ""
    if not reply:
        return jsonify(error="AI အဖြေစာသားမရပါ။ ပြန်စမ်းပါ။"), 502
    return jsonify(reply=reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
