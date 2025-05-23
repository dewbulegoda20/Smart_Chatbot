# app.py - Complete Streamlit Medical Chatbot Application

import streamlit as st
import nltk
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
import warnings
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit page configuration
st.set_page_config(
    page_title="🩺 Medical ChatBot - Dr. AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        margin-left: 20%;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-left: 5px solid #9c27b0;
        margin-right: 20%;
    }
    
    /* Sidebar styling */
    .sidebar-info {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    
    .sidebar-warning {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc02 20%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff9800;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'corpus' not in st.session_state:
        st.session_state.corpus = None
    if 'sentence_list' not in st.session_state:
        st.session_state.sentence_list = []
    if 'article_loaded' not in st.session_state:
        st.session_state.article_loaded = False
    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False

# Download NLTK data (cached)
@st.cache_data
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Load article content (cached)
@st.cache_data
def load_article_content(url):
    """Load and process article content from URL"""
    try:
        # Try to load from the provided URL
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        
        if len(article.text.strip()) < 100:
            raise Exception("Article content too short")
            
        return article.text, True, None
        
    except Exception as e:
        # Fallback content about chronic kidney disease
        fallback_content = """
        Chronic kidney disease (CKD) means your kidneys are damaged and can't filter blood the way they should. 
        The disease is called "chronic" because the damage to your kidneys happens slowly over a long period of time.
        
        The main risk factors for developing kidney disease are diabetes, high blood pressure, heart disease, and a family history of kidney failure.
        People who are older, African American, Hispanic, Native American, or Asian are also at higher risk.
        
        Symptoms of chronic kidney disease develop over time and may include fatigue, swollen ankles, feet, or hands, shortness of breath, blood in urine, foamy urine, and frequent urination especially at night.
        Many people with early kidney disease have no symptoms at all.
        
        Treatment focuses on slowing the progression of kidney damage by controlling the underlying cause.
        This may include medications to control blood pressure, blood sugar levels, and cholesterol.
        Lifestyle changes such as eating a kidney-friendly diet, exercising regularly, not smoking, and limiting alcohol can help.
        
        Early detection and treatment can help slow or prevent the progression of kidney disease.
        Regular monitoring of blood pressure and blood sugar levels is important for people at risk.
        People with diabetes should have their kidneys checked at least once a year.
        
        In advanced stages, treatment options include dialysis or kidney transplant.
        Dialysis is a treatment that filters waste and excess water from the blood when the kidneys can no longer do this effectively.
        A kidney transplant involves surgically placing a healthy kidney from a donor into a person whose kidneys have failed.
        
        Prevention is key and includes maintaining a healthy weight, staying active, eating a balanced diet, managing blood pressure and blood sugar, not smoking, and limiting alcohol consumption.
        """
        return fallback_content, False, str(e)

# Chatbot response functions (from your original code)
def greeting_response(text):
    """Return a random greeting response"""
    text = text.lower()
    bot_greetings = [
        'Hello! 👋 How can I help you today?', 
        'Hi there! 😊 What would you like to know about kidney disease?', 
        'Hey! 🌟 I\'m here to help with your medical questions!', 
        'Greetings! 🤝 Ask me anything about chronic kidney disease!',
        'Nice to meet you! 💫 How may I assist you?'
    ]
    user_greetings = ['hi', 'hey', 'hello', 'hola', 'greetings', 'wassup', 
                     'good morning', 'good afternoon', 'good evening']
    
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)
    return None

def index_sort(list_var):
    """Sort indices based on similarity scores"""
    length = len(list_var)
    list_index = list(range(0, length))
    
    x = list_var
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

def get_bot_response(user_input, sentence_list):
    """Generate bot response using cosine similarity"""
    if not sentence_list:
        return "I don't have any information loaded yet. Please load an article first! 📚"
    
    user_input_lower = user_input.lower()
    sentence_list_copy = sentence_list.copy()
    sentence_list_copy.append(user_input_lower)
    
    bot_response = ''
    
    try:
        # Create count vectorizer and compute similarity
        cm = CountVectorizer().fit_transform(sentence_list_copy)
        similarity_scores = cosine_similarity(cm[-1], cm)
        similarity_scores_list = similarity_scores.flatten()
        
        # Sort by similarity
        index = index_sort(similarity_scores_list)
        index = index[1:]  # Remove the user input itself
        
        response_flag = 0
        j = 0
        
        # Get top 2 most similar sentences
        for i in range(len(index)):
            if similarity_scores_list[index[i]] > 0.0:
                bot_response = bot_response + ' ' + sentence_list_copy[index[i]]
                response_flag = 1
                j = j + 1
            if j >= 2:
                break
        
        # If no similar content found
        if response_flag == 0:
            bot_response = "I apologize, I don't understand that question. Could you please rephrase your question about chronic kidney disease? 🤔 Try asking about symptoms, causes, treatments, or prevention."
            
    except Exception as e:
        bot_response = "I'm having trouble processing your question right now. Please try asking about chronic kidney disease symptoms, causes, treatments, or prevention methods. 🔄"
    
    return bot_response.strip()

# Main application function
def main():
    """Main Streamlit application"""
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Dr. AI - Medical ChatBot</h1>
        <p style="font-size: 1.2em; margin-top: 1rem;">Your intelligent assistant for Chronic Kidney Disease information</p>
        <p style="font-size: 0.9em; opacity: 0.9;">Ask questions about symptoms, causes, treatments, and prevention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        
        # Article loading section
        st.subheader("📖 Knowledge Base")
        
        article_url = st.text_input(
            "Medical Article URL:",
            value="https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521",
            help="Enter the URL of a medical article to use as the knowledge base"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Load Article", type="primary", use_container_width=True):
                with st.spinner("🔍 Loading article content..."):
                    # Download NLTK data first
                    if download_nltk_data():
                        # Load article content
                        corpus, success, error = load_article_content(article_url)
                        
                        if success:
                            st.success("✅ Article loaded successfully!")
                            st.session_state.corpus = corpus
                            st.session_state.sentence_list = nltk.sent_tokenize(corpus)
                            st.session_state.article_loaded = True
                        else:
                            st.warning("⚠️ Using fallback content about kidney disease")
                            st.session_state.corpus = corpus
                            st.session_state.sentence_list = nltk.sent_tokenize(corpus)
                            st.session_state.article_loaded = True
                            if error:
                                st.error(f"Original error: {error}")
                    else:
                        st.error("Failed to download required language data")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.chat_started = False
                st.rerun()
        
        # Status display
        if st.session_state.article_loaded:
            st.markdown("""
            <div class="sidebar-info">
                <strong>📊 System Status</strong><br>
                ✅ Knowledge base loaded<br>
                🤖 ChatBot ready<br>
                💬 Ready for questions
            </div>
            """, unsafe_allow_html=True)
            
            # Display statistics
            if st.session_state.sentence_list:
                st.metric("📝 Sentences Processed", len(st.session_state.sentence_list))
            
            if st.session_state.messages:
                user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
                bot_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
                st.metric("💬 Total Messages", user_msgs + bot_msgs)
        else:
            st.markdown("""
            <div class="sidebar-warning">
                <strong>⚠️ Setup Required</strong><br>
                Please click "Load Article" to initialize the chatbot
            </div>
            """, unsafe_allow_html=True)
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        st.write("Click any question to try it:")
        
        sample_questions = [
            "What are the symptoms of chronic kidney disease?",
            "What causes kidney disease?",
            "How is CKD treated?",
            "What are the risk factors for kidney disease?",
            "How can I prevent kidney disease?",
            "What is dialysis?",
            "When is a kidney transplant needed?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.button(f"❓ {question}", key=f"sample_{i}"):
                if st.session_state.article_loaded:
                    # Add user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    # Get bot response
                    response = get_bot_response(question, st.session_state.sentence_list)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    st.session_state.chat_started = True
                    st.rerun()
                else:
                    st.error("Please load the article first!")
        
        # Help section
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Load Article**: Click "Load Article" button
        2. **Ask Questions**: Type in the chat box below
        3. **Try Samples**: Use sample questions above
        4. **Clear Chat**: Reset conversation anytime
        
        **Tips:**
        - Ask specific questions about symptoms, causes, treatments
        - Use simple, clear language
        - Try rephrasing if you don't get the answer you want
        """)
    
    # Main chat interface
    if not st.session_state.article_loaded:
        st.warning("⚠️ **Please load the knowledge base first!**")
        st.info("👈 Click the '🔄 Load Article' button in the sidebar to get started.")
        st.markdown("---")
        st.markdown("### 🚀 What this ChatBot can do:")
        st.markdown("""
        - Answer questions about **Chronic Kidney Disease**
        - Provide information on **symptoms, causes, and treatments**
        - Explain **prevention methods and risk factors**
        - Discuss **dialysis and transplant options**
        - Give **lifestyle recommendations**
        """)
        return
    
    # Welcome message
    if not st.session_state.chat_started and not st.session_state.messages:
        st.markdown("""
        ### 👋 Welcome to Dr. AI ChatBot!
        
        I'm ready to answer your questions about **Chronic Kidney Disease**. You can:
        
        - 💬 Type your questions in the chat box below
        - 🎯 Click on sample questions in the sidebar
        - 🔍 Ask about symptoms, causes, treatments, or prevention
        
        **Go ahead, ask me anything about kidney health!** 👇
        """)
    
    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display all chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🩺 Dr. AI ({message['timestamp']}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input form
    st.markdown("---")
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "💬 Ask your question:",
                placeholder="Type your question about chronic kidney disease here...",
                key="user_input",
                help="Ask about symptoms, causes, treatments, prevention, or any aspect of kidney disease"
            )
        
        with col2:
            send_button = st.form_submit_button("📤 Send", type="primary", use_container_width=True)
    
    # Process user input
    if send_button and user_input and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Check for greeting first
        greeting_resp = greeting_response(user_input)
        if greeting_resp:
            response = greeting_resp
        else:
            # Get bot response using similarity matching
            with st.spinner("🤔 Thinking..."):
                response = get_bot_response(user_input, st.session_state.sentence_list)
        
        # Add bot response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        st.session_state.chat_started = True
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🩺 Dr. AI ChatBot - Your Medical Information Assistant</p>
        <p style="font-size: 0.8em;">⚠️ This chatbot provides general information only. Always consult healthcare professionals for medical advice.</p>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()