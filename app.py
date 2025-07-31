import eventlet
eventlet.monkey_patch()

import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sessions'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# --- AI PERSONA AND PROMPT CONFIG ---
resume_summary = """
Mithun Raj M R, aspiring software developer. Proficient in Python, Flask, React, MySQL, ML (ASL detection), 
internships at AuMDS (led full-stack projects), Varcons (web dev), and strong problem-solving skills.
"""
template = """
You are Mithun Raj M R, a software developer. Your task is to answer interview questions by fully embodying this persona.
**RULES:**
- Speak in the first person using "I", "my", and "me".
- NEVER say "As Mithun" or "Here is my response". You ARE Mithun.
- Keep answers professional, confident, and concise (2-4 sentences).
- When asked for code, present it naturally. For example: "Certainly. I would solve that using the Euclidean algorithm. Here's how I'd write it:"
- Always generate code in Python unless the user explicitly asks for a different language.
- Base all technical and project-related answers on the summary below.

**Conversation History (most recent turns):**
{context}

**Interviewer's Question:** {question}

**Your Answer (as Mithun):**
"""
prompt_template = ChatPromptTemplate.from_template(template)
model = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

# --- State Management ---
user_contexts = {}
user_busy_state = {}

@app.route('/')
def index():
    return render_template('index.html')

# --- SOCKET.IO EVENT HANDLERS ---
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('start_session')
def handle_start_session():
    user_contexts[request.sid] = []
    user_busy_state[request.sid] = False
    print(f"Session started for {request.sid}")

@socketio.on('user_sends_message')
def handle_user_message(data):
    transcript = data.get('transcript')
    sid = request.sid
    if user_busy_state.get(sid, False):
        print(f"Request from {sid} ignored, AI is busy.")
        socketio.emit('ai_is_busy', to=sid)
        return
    if not transcript: return
    print(f"Received transcript from {sid}: {transcript}")
    socketio.start_background_task(stream_ai_response, transcript, sid)

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    print(f"Client disconnected: {sid}")
    if sid in user_contexts: del user_contexts[sid]
    if sid in user_busy_state: del user_busy_state[sid]

# --- AI STREAMING FUNCTION ---
def stream_ai_response(user_question, sid):
    user_busy_state[sid] = True
    try:
        history = user_contexts.get(sid, [])
        recent_history = history[-5:]
        formatted_context = "\n".join([f"Interviewer: {q}\nMithun: {a}" for q, a in recent_history])
        
        formatted_prompt = prompt_template.format(
            resume_summary=resume_summary,
            context=formatted_context,
            question=user_question
        )
        
        socketio.emit('ai_stream_start', to=sid)
        full_ai_response = ""
        for token in model.stream(formatted_prompt):
            content = token.content
            socketio.emit('ai_stream', content, to=sid)
            full_ai_response += content
            
        history.append((user_question, full_ai_response))
        user_contexts[sid] = history[-5:]
        print(f"[AI] Full response generated for {sid}")

    except Exception as e:
        print(f"An error occurred while streaming: {e}")
    finally:
        socketio.emit('ai_stream_end', to=sid)
        user_busy_state[sid] = False
        print(f"Session lock released for {sid}")

# --- MAIN EXECUTION (for local testing) ---
if __name__ == '__main__':
    print("Starting Flask-SocketIO server for local development...")
    socketio.run(app, host='0.0.0.0', port=5000)