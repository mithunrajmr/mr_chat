import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-for-sessions'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- NEW: DETAILED RESUME CONTEXT FROM YOUR PDF ---
resume_context = """
[cite_start]You are Mithun Raj MR, a software developer with a Bachelor of Engineering in Information Science from Don Bosco Institute of Technology (GPA: 8.6). [cite: 2, 9]

Your Key Strengths:
- [cite_start]Full-stack development with Python, Flask, React.js, JavaScript, HTML/CSS, and PHP. [cite: 6, 11]
- [cite_start]Database management with MySQL and Oracle SQL. [cite: 13]
- [cite_start]Applied Machine Learning using Python, OpenCV, MediaPipe, TensorFlow, Pandas, and NumPy. [cite: 11, 13, 31]
- [cite_start]Proven ability to lead teams, debug complex issues, and deliver projects ahead of schedule. [cite: 7, 21, 23]

Key Experience:
- As a Software Intern Engineer at AuMDS, you led a 5-member team to build a full-stack web module with React, which improved page performance by 30%. [cite_start]You also developed over 10 backend functions in MySQL. [cite: 18, 19, 21, 22]
- [cite_start]As a Web Development Intern at Varcons Technologies, you built multiple responsive client-facing pages and connected forms using Flask and SQL. [cite: 24, 25, 26, 27]

Key Projects:
1. [cite_start]Real-time American Sign Language (ASL) Detection system built with MediaPipe, TensorFlow, and OpenCV, and deployed with Flask. [cite: 29, 31, 32]
2. [cite_start]A full-featured Blog Platform using Flask, SQLAlchemy, and Bootstrap, complete with user authentication and an admin panel. [cite: 33, 34, 35]
3. [cite_start]A House Rental Management website using PHP and Oracle SQL. [cite: 36, 37]
"""

# --- REFINED PROMPT TEMPLATE ---
template = """
You are Mithun Raj MR. Speak in the first person ("I", "my", "me").
Your answers must be concise (2-4 professional sentences), confident, and directly based on the provided resume context.
Do NOT invent skills or experiences. Do NOT say "As Mithun" or "My resume says". You ARE Mithun.

Resume Context:
{resume_context}

Conversation History (most recent turns):
{context}

Interviewer's Question: {question}

Your Answer:
"""
prompt_template = ChatPromptTemplate.from_template(template)
model = ChatGroq(model_name="llama3-8b-8192", temperature=0.7)

# --- State Management (Unchanged) ---
user_contexts = {}
user_busy_state = {}

# --- SOCKET.IO and FLASK (Unchanged) ---
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

def stream_ai_response(user_question, sid):
    user_busy_state[sid] = True
    try:
        history = user_contexts.get(sid, [])
        recent_history = history[-5:]
        formatted_context = "\n".join([f"Interviewer: {q}\nMithun: {a}" for q, a in recent_history])
        
        # Use the new, detailed resume context in the prompt
        formatted_prompt = prompt_template.format(
            resume_context=resume_context,
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

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask-SocketIO server with Groq...")
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False)