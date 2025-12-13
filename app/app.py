import streamlit as st
import requests
import json
import joblib
import numpy as np
import os
import time
from groq import Groq
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime



# --- CONFIGURATION ---
# Load API Keys (Best practice: use st.secrets for deployment, or os.getenv locally)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
PROMPT_FILE = "prompt.txt"

# 1. Get the absolute path of the folder where app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the paths safely
PROMPT_FILE = os.path.join(current_dir, "..", "lab", "prompt.txt")
ECO_MODEL_FILE = os.path.join(current_dir, "..", "bin", "model_economic.pkl")
SOC_MODEL_FILE = os.path.join(current_dir, "..", "bin", "model_social.pkl")

# --- 1. CACHED RESOURCES (Load once for speed) ---

@st.cache_resource
def load_models():
    """Loads the ML predictors and Embedding model."""
    try:
        mod_soc = joblib.load(SOC_MODEL_FILE)
        mod_eco = joblib.load(ECO_MODEL_FILE)
        # Load embedding model (downloads if not present)
        embedder = SentenceTransformer('all-MiniLM-L6-v2') 
        return mod_soc, mod_eco, embedder
    except Exception as e:
        st.error(f"Failed to load models: {e}. Did you run train_real_data.py?")
        return None, None, None

@st.cache_data
def load_prompt():
    with open(PROMPT_FILE, "r") as f:
        return f.read()

# Initialize Resources
model_social, model_economic, embedder = load_models()
prompt_template = load_prompt()

# Anchor concepts for semantic filtering (Must match training!)
ANCHOR_TEXTS = [
    "Civil unrest riots protests violence and police clashes",
    "Economic crisis inflation stock market crash and unemployment"
]
if embedder:
    anchor_embeddings = embedder.encode(ANCHOR_TEXTS, convert_to_tensor=True)

# --- 2. BACKEND FUNCTIONS ---

def fetch_live_news():
    """Fetches top 50 headlines from the US (or your target country)."""
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=50&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("status") != "ok":
            st.error(f"NewsAPI Error: {data.get('message')}")
            return []
        
        # Combine Title + Description
        articles = [
            f"{a['title']}. {a['description'] or ''}" 
            for a in data.get("articles", [])
        ]
        return articles
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

def semantic_filter(articles, top_k=20):
    """Filters live news to keep only the most risk-relevant items."""
    if not articles: return ""
    
    # Embed live articles
    article_embeddings = embedder.encode(articles, convert_to_tensor=True)
    
    # Compare to anchors
    cosine_scores = util.cos_sim(article_embeddings, anchor_embeddings)
    max_scores, _ = torch.max(cosine_scores, dim=1)
    
    # Get Top K
    top_results = torch.topk(max_scores, k=min(top_k, len(articles)))
    selected_indices = top_results.indices.tolist()
    
    selected_news = [articles[i] for i in selected_indices]
    return ". ".join(selected_news)

def analyze_with_groq(news_text):
    """Sends filtered news to Groq for scoring."""
    client = Groq(api_key=GROQ_API_KEY)
    
    final_prompt = prompt_template.replace("{text}", news_text)
    
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": final_prompt}],
            model="llama-3.1-8b-instant", # The NEW supported model
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Groq API Error: {e}")
        return {"social_score": 0, "economic_score": 0}

# --- 3. FRONTEND (DASHBOARD) ---

st.set_page_config(page_title="Sentinell: AI Risk Monitor", layout="wide")

# Header
st.title("🌍 Sentinell: AI Socio-Economic Risk Monitor")
st.markdown(f"**Live Analysis for:** {datetime.now().strftime('%A, %d %B %Y')}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🔄 Run Live Analysis", type="primary"):
        with st.spinner("1. Fetching Global News..."):
            raw_news = fetch_live_news()
            
        with st.spinner("2. Filtering for Risk Signals (Semantic Search)..."):
            filtered_news = semantic_filter(raw_news)
            
        with st.spinner("3. Consult Llama 3.1 (Groq)..."):
            scores = analyze_with_groq(filtered_news)
            
        with st.spinner("4. Running Predictive Models..."):
            # Prepare input [Social, Economic]
            X_input = np.array([[scores['social_score'], scores['economic_score']]])
            
            # Predict Probabilities (we want the probability of class "1")
            prob_social = model_social.predict_proba(X_input)[0][1]
            prob_economic = model_economic.predict_proba(X_input)[0][1]
            
        # Store in session state to keep results on screen
        st.session_state['data'] = {
            'scores': scores,
            'probs': {'social': prob_social, 'econ': prob_economic},
            'news_preview': filtered_news
        }
    
    st.divider()
    st.info("System uses **Llama-3.1-8b** for anxiety scoring and **Logistic Regression** for event prediction.")

# Display Results
if 'data' in st.session_state:
    data = st.session_state['data']
    scores = data['scores']
    probs = data['probs']
    
    # --- ROW 1: LLM ANXIETY SCORES ---
    st.subheader("🧠 LLM Estimated Anxiety Levels (0-10)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Social Anxiety Score", f"{scores['social_score']}/10")
        st.progress(scores['social_score'] / 10)
        
    with col2:
        st.metric("Economic Anxiety Score", f"{scores['economic_score']}/10")
        st.progress(scores['economic_score'] / 10)
        
    st.divider()
    
    # --- ROW 2: PREDICTIVE MODEL ODDS ---
    st.subheader("🔮 30-Day Crisis Forecast (ML Probability)")
    p_col1, p_col2 = st.columns(2)
    
    def get_color(prob):
        return "green" if prob < 0.3 else "orange" if prob < 0.7 else "red"
    
    with p_col1:
        st.markdown("### 🚨 Social Unrest Risk")
        val = probs['social']
        color = get_color(val)
        st.markdown(f"<h1 style='color:{color}'>{val:.1%}</h1>", unsafe_allow_html=True)
        if val > 0.5:
            st.warning("⚠️ High Risk of Civil Disorder Detected")
        else:
            st.success("✅ Stability Forecasted")

    with p_col2:
        st.markdown("### 📉 Economic Crisis Risk")
        val = probs['econ']
        color = get_color(val)
        st.markdown(f"<h1 style='color:{color}'>{val:.1%}</h1>", unsafe_allow_html=True)
        if val > 0.5:
            st.warning("⚠️ High Risk of Financial Stress Detected")
        else:
            st.success("✅ Markets Appear Stable")

    # --- ROW 3: TRANSPARENCY ---
    with st.expander("🔍 View Analyzed News Content (Evidence)"):
        st.write(data['news_preview'])
        
else:
    st.info("👈 Click **'Run Live Analysis'** in the sidebar to start.")
    

### run using : streamlit run app/app.py