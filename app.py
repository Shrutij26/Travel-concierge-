import streamlit as st
import os
import requests
from dotenv import load_dotenv

from multi_agent import build_multi_agent_graph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from auth_db import init_db, create_user, authenticate_user, get_user_quota, increment_user_quota, get_user_preferences, update_user_preference
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

import folium
from streamlit_folium import st_folium
from fpdf import FPDF
import re

from advanced_rag import get_travel_knowledge

# Load environment variables
load_dotenv()

# --- Tools Implementation ---

def get_weather(city: str) -> str:
    """Returns current weather and temperature for a given city."""
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        return "Error: OPENWEATHER_API_KEY is not set in environment variables."
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The current weather in {city} is {desc} with a temperature of {temp}°C."
    else:
        return f"Could not fetch weather for {city}. Please check the city name."

def get_top_attractions(city: str) -> str:
    """Returns top 5 tourist attractions with name and description for a given city."""
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    if not api_key:
        return "Error: GEOAPIFY_API_KEY is not set in environment variables."
    
    # 1. Get coordinates for the city
    geo_url = f"https://api.geoapify.com/v1/geocode/search?text={city}&format=json&apiKey={api_key}"
    geo_response = requests.get(geo_url)
    if geo_response.status_code != 200:
        return f"Could not find coordinates for {city}."
    
    geo_data = geo_response.json()
    if not geo_data.get("results"):
        return f"Could not find coordinates for {city}."
        
    lat = geo_data["results"][0]["lat"]
    lon = geo_data["results"][0]["lon"]
    
    # 2. Get top attractions around these coordinates
    places_url = f"https://api.geoapify.com/v2/places?categories=tourism.sights&filter=circle:{lon},{lat},10000&bias=proximity:{lon},{lat}&limit=5&apiKey={api_key}"
    places_response = requests.get(places_url)
    
    if places_response.status_code == 200:
        places_data = places_response.json()
        places = places_data.get("features", [])
        if not places:
            return f"No attractions found near {city}."
            
        result = f"Top attractions in {city}:\n"
        for i, place in enumerate(places):
            props = place.get("properties", {})
            name = props.get("name", "Unknown Attraction")
            categories = ", ".join([c.split('.')[-1] for c in props.get("categories", []) if c.startswith("tourism")])
            result += f"{i+1}. {name} (Categories: {categories.title()})\n"
        return result
    else:
        return f"Could not fetch attractions for {city}."

def get_location_map(query: str) -> str:
    """Useful to get the map coordinates for a hotel, place, or city when the user asks for its location."""
    api_key = os.environ.get("GEOAPIFY_API_KEY")
    if not api_key: return "Error: API key missing."
    geo_url = f"https://api.geoapify.com/v1/geocode/search?text={query}&format=json&apiKey={api_key}"
    resp = requests.get(geo_url)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("results"):
            lat = data["results"][0]["lat"]
            lon = data["results"][0]["lon"]
            name = data["results"][0].get("formatted", query)
            st.session_state["map_data"] = {
                "city": name,
                "center_lat": lat,
                "center_lon": lon,
                "places": [{"name": name, "lat": lat, "lon": lon}]
            }
            return f"Successfully generated a map for {name}. The user can now see it."
    return f"Could not find coordinates for {query}."

def save_user_preference(pref_string: str) -> str:
    """Useful to save user preferences, facts, dietary restrictions, budget styles, and long-term memory for future sessions.
    Input should be exactly in the format 'key:value' (e.g. 'diet:vegetarian', 'favorite_city:Paris')."""
    if "user_id" not in st.session_state or not st.session_state.user_id:
        return "Error: User not logged in."
    
    try:
        key, value = pref_string.split(":", 1)
        update_user_preference(st.session_state.user_id, key.strip(), value.strip())
        return f"Successfully saved preference '{key.strip()}' as '{value.strip()}' into semantic memory."
    except ValueError:
        return "Error: Input must be exactly in 'key:value' format."

# --- App Setup ---

st.set_page_config(page_title="Maya — Your Travel Concierge", page_icon="✈️", layout="wide")

init_db()
MAX_API_CALLS = 10

if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.username = None

def login_signup_ui():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>✈️ Welcome to Maya AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your personal, intelligent travel planner.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            l_user = st.text_input("Username", key="l_user")
            l_pass = st.text_input("Password", type="password", key="l_pass")
            if st.button("Login", key="l_btn", use_container_width=True):
                uid = authenticate_user(l_user, l_pass)
                if uid:
                    st.session_state.user_id = uid
                    st.session_state.username = l_user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab2:
            s_user = st.text_input("New Username", key="s_user")
            s_pass = st.text_input("New Password", type="password", key="s_pass")
            if st.button("Sign Up", key="s_btn", use_container_width=True):
                if create_user(s_user, s_pass):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Username already exists.")

if not st.session_state.user_id:
    login_signup_ui()
    st.stop()

@st.cache_resource
def get_checkpointer():
    conn = sqlite3.connect("travel_checkpoints.db", check_same_thread=False)
    # create LangGraph checkpointer tables if they don't exist
    return SqliteSaver(conn)

# Custom CSS for a beautiful, premium aesthetic
st.markdown("""
<style>
    .hero-container {
        background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), url("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?auto=format&fit=crop&w=2000&q=80");
        background-size: cover;
        background-position: center;
        padding: 50px 20px;
        border-radius: 20px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    .hero-title {
        color: white !important;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        color: #f8f9fa !important;
        font-size: 1.3rem !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .stButton>button {
        border-radius: 20px;
        border: 1px solid #DD2476;
        color: #DD2476;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #DD2476;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Add the hero section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">✈️ Maya AI Concierge</h1>
    <p class="hero-subtitle">Your personal, intelligent travel planner.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title(f"✈️ Hello, {st.session_state.username}!")
quota = get_user_quota(st.session_state.user_id)
st.sidebar.markdown(f"**API Quota:** {quota}/{MAX_API_CALLS}")
if st.sidebar.button("Logout"):
    st.session_state.user_id = None
    st.session_state.username = None
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.image("https://images.unsplash.com/photo-1436491865332-7a61a109cc05?q=80&w=800&auto=format&fit=crop", use_container_width=True)
st.sidebar.markdown("### Active Tools:")
st.sidebar.markdown("- 🌤️ **Weather**: OpenWeatherMap")
st.sidebar.markdown("- 🏛️ **Places**: Geoapify")
st.sidebar.markdown("- 🔍 **Web Search**: Tavily")
st.sidebar.markdown("- 🧠 **RAG KB**: Advanced RAG")

st.markdown("---")
st.markdown("### 🎯 Customize Your Trip")
col1, col2, col3, col4 = st.columns(4)
with col1:
    budget = st.slider("💰 Budget (₹)", min_value=5000, max_value=200000, step=1000, value=25000, key="budget")
with col2:
    travel_style = st.selectbox("👥 Traveling As", ["Solo", "Couple", "Family", "Friends"], key="style")
with col3:
    duration = st.selectbox("⏱️ Duration", ["1-3 Days", "4-7 Days", "1-2 Weeks", "1 Month+"], key="duration")
with col4:
    interests = st.selectbox("🎨 Interests", ["Adventure", "Relaxation", "Culture", "Food & Nightlife"], key="interests")
st.markdown("---")

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    # Note: For SQLite checkpointer, true clearing would involve deleting the thread. 
    # For now, we just clear the local UI.
    if "map_data" in st.session_state:
        del st.session_state["map_data"]
    st.rerun()

# FEATURE 3: Download trip as PDF (Always visible in sidebar)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 Export Trip")

# Get last assistant message or provide a default text if chat is empty
if "messages" in st.session_state and any(m["role"] == "assistant" for m in st.session_state.messages):
    last_assistant_msg = [m["content"] for m in st.session_state.messages if m["role"] == "assistant"][-1]
else:
    last_assistant_msg = "No trip planned yet! Ask Maya to plan a trip for you."

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    # Strip complex emojis and format HTML tags for basic text
    clean_text = re.sub(r'<[^>]+>', '', text) # remove HTML
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1') # Replace unsupported chars with ?
    pdf.multi_cell(0, 8, text=clean_text)
    return bytes(pdf.output()) # FIX: cast bytearray to bytes

pdf_bytes = create_pdf(last_assistant_msg)
st.sidebar.download_button(
    label="📥 Download Trip Plan (PDF)",
    data=pdf_bytes,
    file_name="Maya_Trip_Plan.pdf",
    mime="application/pdf"
)

# Initialize session state for memory and messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "memory" not in st.session_state:
    st.session_state.memory = get_checkpointer()

# Enable LangSmith Tracing if API key exists
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Travel-Concierge-MultiAgent"

# --- Tool Guardrails ---
def safe_tool_wrapper(func):
    """Wrapper to handle tool hallucinations or errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f"Tool execution failed: {str(e)}. Please correct your arguments and try again."
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# Define Specialized Tools for Sub-Agents
local_guide_tools = [
    Tool(
        name="AttractionsTool",
        func=safe_tool_wrapper(get_top_attractions),
        description="Useful for finding the top tourist attractions and places to visit in a specific city. Input should be the city name."
    ),
    Tool(
        name="TravelKnowledgeTool",
        func=safe_tool_wrapper(get_travel_knowledge),
        description="Useful to search curated travel knowledge base like PDF guides and blogs. Use this to find specialized travel tips, best time to visit, and budget options based on curated travel data."
    )
]

logistics_tools = [
    Tool(
        name="WeatherTool",
        func=safe_tool_wrapper(get_weather),
        description="Useful for getting the current weather and temperature for a specific city. Input should be the city name."
    ),
    Tool(
        name="LocationMapTool",
        func=safe_tool_wrapper(get_location_map),
        description="Useful for finding the exact location of a hotel, restaurant, or place and displaying it on a map. Use this ONLY when the user asks to see the location or asks for hotel suggestions."
    )
]

budget_tools = [
    TavilySearchResults(
        max_results=3, 
        include_images=True,
        description="Useful for searching the web for budget tips, best time to visit, local food recommendations, and finding beautiful images of places."
    )
]

planner_tools = [
    Tool(
        name="SavePreferenceTool",
        func=safe_tool_wrapper(save_user_preference),
        description="Useful to save the user's long-term preferences to Semantic Memory, like their diet, budget style, or favorite places. Input format: 'key:value' (e.g. 'diet:vegan'). Use this whenever the user states a preference."
    )
]

tools_dict = {
    "local_guide_tools": local_guide_tools,
    "logistics_tools": logistics_tools,
    "budget_tools": budget_tools,
    "planner_tools": planner_tools
}

# Initialize Agent Graph
if not os.environ.get("GROQ_API_KEY"):
    st.warning("Please add your GROQ_API_KEY to the .env file to run the app.")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

agent = build_multi_agent_graph(
    llm=llm,
    tools_dict=tools_dict,
    checkpointer=st.session_state.memory
)

st.markdown("### 💬 Chat with Maya")
# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# FEATURE 4: Map embed render
if "map_data" in st.session_state:
    st.markdown("---")
    st.markdown(f"### 🗺️ Map: {st.session_state['map_data']['city']}")
    try:
        lat = float(st.session_state["map_data"]["center_lat"])
        lon = float(st.session_state["map_data"]["center_lon"])
        m = folium.Map(location=[lat, lon], zoom_start=14)
        for place in st.session_state["map_data"]["places"]:
            p_lat = float(place["lat"])
            p_lon = float(place["lon"])
            folium.Marker([p_lat, p_lon], popup=place["name"], tooltip=place["name"]).add_to(m)
        st_folium(m, height=400, use_container_width=True, key="dynamic_map")
    except Exception as e:
        st.error(f"Could not render map correctly. Please try asking again. Details: {str(e)}")

# FEATURE 1: Suggestion chips (Always visible)
st.markdown("---")
st.markdown("**💡 Quick Suggestions:**")
chip_prompt = None
cols = st.columns(4)
if cols[0].button("🎒 Show packing list"):
    chip_prompt = "Show packing list"
if cols[1].button("💰 Best budget tips"):
    chip_prompt = "What are the best budget tips?"
if cols[2].button("🌤️ When to visit?"):
    chip_prompt = "When is the best time to visit?"
if cols[3].button("🏨 Find top hotels"):
    chip_prompt = "Find top hotels"

# Chat input
chat_prompt = st.chat_input("Ask Maya about a destination...")
prompt = chat_prompt or chip_prompt

if prompt:
    # Clear map data on new question
    if "map_data" in st.session_state:
        del st.session_state["map_data"]

    # FEATURE 2 & Semantic Memory: Budget-aware planning (Inject all filters and DB preferences into prompt)
    prefs = get_user_preferences(st.session_state.user_id)
    prefs_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()]) if prefs else "None yet."
    enhanced_prompt = f"[System Note: Budget: ₹{budget}. Traveling as: {travel_style}. Duration: {duration}. Interests: {interests}. Saved Long-term Preferences (Semantic Memory): {prefs_str}] {prompt}"
    
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        quota = get_user_quota(st.session_state.user_id)
        if quota >= MAX_API_CALLS:
            st.error("⚠️ You have reached your API call limit. Please upgrade your plan.")
        else:
            try:
                increment_user_quota(st.session_state.user_id)
                config = {"configurable": {"thread_id": f"maya_session_{st.session_state.user_id}"}}
                
                with st.status("Maya is thinking... ✈️", expanded=True) as status:
                    # Use stream to show progress and reduce perceived latency
                    for event in agent.stream({"messages": [{"role": "user", "content": enhanced_prompt}]}, config=config, stream_mode="updates"):
                        for node, data in event.items():
                            if node == "Planner":
                                status.update(label="Planner is analyzing and orchestrating...", state="running")
                            elif node in ["LocalGuideAgent", "LogisticsAgent", "BudgetAgent"]:
                                status.update(label=f"{node} is gathering specific data...", state="running")
                                if "messages" in data and len(data["messages"]) > 0:
                                    st.write(f"⚙️ **{node}**: Completed task.")
                    
                    status.update(label="Response ready!", state="complete", expanded=False)
                
                # Fetch the final message from state
                state = agent.get_state(config)
                # The final answer is the last message in the sequence
                raw_response = state.values["messages"][-1].content
                
                # Fix Gemini dict list format
                if isinstance(raw_response, list):
                    text_parts = [part["text"] for part in raw_response if isinstance(part, dict) and "text" in part]
                    response = "\n".join(text_parts)
                else:
                    response = str(raw_response)
                    
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Rerun to cleanly render the messages above the suggestion chips
    st.rerun()
