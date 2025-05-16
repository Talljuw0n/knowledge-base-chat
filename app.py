import os
import fitz  # PyMuPDF
import docx
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import uuid
import json

# For LLM integration
from groq import Groq

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY", "default_secret_key")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

# Store session documents and text - in a real app, use a database
user_documents = {}

# Store conversation history for each session
conversation_history = {}

# Store last question for quiz tracking
last_quiz_questions = {}
last_quiz_answers = {}  # Store the correct answers

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_text_from_pdf(filepath):
    """Extract text content from PDF files."""
    text = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        text = f"Error processing PDF: {str(e)}"
    return text


def extract_text_from_docx(filepath):
    """Extract text content from DOCX files."""
    try:
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return f"Error processing DOCX: {str(e)}"


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle document uploads and text extraction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    session_id = request.form.get("session_id", str(uuid.uuid4()))

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text based on file type
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.docx'):
            extracted_text = extract_text_from_docx(filepath)
        elif filename.lower().endswith('.txt'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='latin-1') as f:
                        extracted_text = f.read()
                except Exception as e:
                    extracted_text = f"Error reading text file: {str(e)}"
        else:
            extracted_text = "Unsupported file format. Please upload PDF, DOCX, or TXT files."

        # Store the extracted text
        user_documents[session_id] = extracted_text
        
        # Return success response with session_id
        return jsonify({
            'message': 'File uploaded and text extracted successfully.',
            'session_id': session_id,
            'filename': filename
        })
    
    return jsonify({'error': 'File upload failed'}), 500


def build_prompt(user_input, context, session_id):
    """Build the prompt for the LLM based on user input and conversation history."""
    user_input_lower = user_input.strip().lower()
    
    # Get conversation history for this session
    session_history = conversation_history.get(session_id, [])
    
    if user_input_lower == "quiz me" or user_input_lower == "another question":
        prompt = f"""
        Based on the content below, generate ONE multiple-choice quiz question with options A, B, C, and D. 
        The question should test understanding of key concepts in the document.
        
        Make sure the question and answers are factually accurate and directly based on the document content.
        Don't make up information that isn't in the document.
        
        Format your response like this:
        QUESTION: [Your question here]
        A) [Option A]
        B) [Option B]
        C) [Option C]
        D) [Option D]
        
        CORRECT_ANSWER: [letter]
        EXPLANATION: [Brief explanation of why the answer is correct]

        Here is the document content:
        {context}
        """
        return prompt, "quiz"
        
    elif user_input_lower in ["a", "b", "c", "d"]:
        # Check if we have a last question for this session
        last_question = last_quiz_questions.get(session_id)
        correct_answer = last_quiz_answers.get(session_id)
        
        if not last_question or not correct_answer:
            return "I don't have a record of asking you a question. Please type 'quiz me' first.", "answer_without_question"
            
        prompt = f"""
        The user was asked this multiple-choice question:
        {last_question}
        
        The user answered: {user_input.upper()}
        The correct answer is: {correct_answer}
        
        Evaluate if the answer is correct or incorrect, and explain why the correct answer is right.
        Start your response with "CORRECT" if the user chose {correct_answer}, or "INCORRECT" if they chose a different option.
        Then provide your explanation using markdown formatting to make it visually appealing.
        
        Use **bold** for important concepts, organize with bullet points where appropriate,
        and use headings (## or ###) to structure your response.
        """
        return prompt, "answer"
        
    elif any(phrase in user_input_lower for phrase in ["explain", "summarize", "what's it about", "what is it about", "tell me about"]):
        prompt = f"""
        Please explain the content of the document as if you're talking to a 10-year-old child. 
        Break down complex concepts into very simple ideas using everyday language and familiar examples.
        
        Your explanation should:
        
        1. Start with a friendly introduction that explains the main idea in 1-2 sentences
        2. Break down each important topic into "bite-sized" chunks that are easy to understand
        3. Use lots of examples and comparisons to things a child would know about
        4. Avoid technical jargon completely - if you must use a technical term, explain it immediately
        5. Use short sentences and simple vocabulary
        6. Ask occasional questions to keep the explanation engaging
        7. Use emoji when appropriate to make the content more engaging
        8. End with a very simple summary of the key points
        
        Make your explanation friendly, curious, and enthusiastic - like you're a fun teacher!
        Format using markdown with appropriate headings, bold text, and bullet points to make it visually organized.

        Here is the document content:
        {context}
        """
        return prompt, "explain"
        
    else:
        # Include relevant conversation history for context
        history_text = ""
        if session_history:
            # Get the last 3 exchanges maximum
            recent_history = session_history[-6:]  # 3 exchanges = 6 messages (user + assistant)
            history_text = "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" for i, msg in enumerate(recent_history)])
            history_text = f"\nRecent conversation:\n{history_text}\n"
            
        prompt = f"""
        Based on the document content, answer the following user question.
        Provide a comprehensive and accurate answer based solely on the information in the document.
        Explain the concepts very simply, as if talking to someone who is completely new to the topic.
        Use simple language, short sentences, and avoid technical jargon when possible.
        
        Document:
        {context}
        {history_text}
        Question: {user_input}
        """
        return prompt, "general"


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print("Client connected")
    emit('response', {'message': 'Connected to server. Upload a document to get started.'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")


@socketio.on('message')
def handle_message(data):
    """Process incoming messages and generate responses."""
    session_id = data.get("session_id")
    user_input = data.get("message")
    
    print(f"Received message: {user_input} from session: {session_id}")
    
    # Get document context for this session
    context = user_documents.get(session_id, "")
    
    if not context:
        emit('response', {
            'message': "No documents found for this session. Please upload a document first."
        })
        return
    
    if not user_input:
        emit('response', {'message': "Please enter a question or command."})
        return
    
    # Add message to conversation history
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append(user_input)
    
    try:
        # Special handling for direct quiz request - simulate the session ID for direct chat
        if user_input.lower() == "could you quiz me?":
            direct_quiz_session = "direct_chat_session"
            if direct_quiz_session not in user_documents:
                user_documents[direct_quiz_session] = context
            session_id = direct_quiz_session
            user_input = "quiz me"
        
        # Special handling for single-letter answers in direct chat
        if user_input.lower() in ["a", "b", "c", "d"] and (not last_quiz_questions.get(session_id) and session_id != "direct_chat_session"):
            # Check if we have a last question in the direct chat session
            if "direct_chat_session" in last_quiz_questions:
                session_id = "direct_chat_session"
        
        # Build prompt for the LLM
        prompt, query_type = build_prompt(user_input, context, session_id)
        
        if query_type == "answer_without_question":
            # Handle the case where user answered without a question
            emit('response', {'message': prompt})  # prompt contains the error message
            return
        
        # System prompt based on query type
        system_prompt = "You are a helpful assistant that answers questions in simple, easy-to-understand language."
        if query_type == "explain":
            system_prompt = """You are a friendly teacher who explains complex topics to children.
            You break down difficult concepts into simple, everyday language a 10-year-old would understand.
            You use relatable examples, simple vocabulary, short sentences, and an enthusiastic, warm tone.
            You avoid technical jargon completely unless you immediately explain it in very simple terms."""
        elif query_type == "quiz":
            system_prompt = """You are an educational assessment expert that creates factually accurate quiz questions
            based only on the provided document content."""
            
        # Call Groq API
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Use appropriate model ID
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,  # Increased token limit for more comprehensive responses
            temperature=0.7,  # Slightly increased temperature for more creative explanations
        )
        
        # Extract the response
        ai_response = completion.choices[0].message.content
        
        # Process the response based on query type
        if query_type == "quiz":
            # Parse the quiz response
            question_parts = ai_response.split("CORRECT_ANSWER:")
            
            if len(question_parts) > 1:
                # Extract question and correct answer
                question_text = question_parts[0].strip()
                answer_parts = question_parts[1].strip().split("EXPLANATION:", 1)
                
                correct_answer = answer_parts[0].strip()
                explanation = ""
                if len(answer_parts) > 1:
                    explanation = answer_parts[1].strip()
                
                # Format the question with markdown for better appearance
                formatted_question = question_text
                
                if formatted_question.startswith("QUESTION:"):
                    # Replace the plain "QUESTION:" with a markdown formatted version
                    formatted_question = "## " + formatted_question.replace("QUESTION:", "Quiz Question:", 1)
                
                # Store the quiz question and correct answer
                last_quiz_questions[session_id] = question_text
                last_quiz_answers[session_id] = correct_answer
                
                # Store explanation if available
                if explanation:
                    last_quiz_questions[session_id] += f"\n\nEXPLANATION: {explanation}"
                
                # Send only the formatted question part to the user
                emit('response', {'message': formatted_question})
            else:
                # Fallback if parsing fails
                emit('response', {'message': "I couldn't generate a proper quiz question. Please try again."})
        
        elif query_type == "answer":
            # Format the answer evaluation
            if ai_response.startswith("CORRECT") or ai_response.startswith("INCORRECT"):
                response_parts = ai_response.split("\n", 1)
                if len(response_parts) > 1:
                    result = response_parts[0].strip()
                    explanation = response_parts[1].strip()
                    
                    if result == "CORRECT":
                        formatted_response = f"## ✅ **{result}**\n\n{explanation}"
                    else:
                        formatted_response = f"## ❌ **{result}**\n\n{explanation}"
                    
                    emit('response', {'message': formatted_response})
                else:
                    emit('response', {'message': ai_response})
            else:
                emit('response', {'message': ai_response})
        
        else:
            # For general queries and explanations
            emit('response', {'message': ai_response})
        
        # Add response to conversation history
        conversation_history[session_id].append(ai_response)
        
    except Exception as e:
        print(f"Error generating response: {e}")
        emit('response', {'message': f"Sorry, I encountered an error: {str(e)}"})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)