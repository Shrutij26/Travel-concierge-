\ ✈️ Maya — AI Travel Concierge

> *"Not just another chatbot. Maya is your personal travel buddy powered by real AI."*

Built with **LangChain Agents**, live APIs, and a beautiful Streamlit UI — Maya plans your entire trip, fetches real-time data, and even exports your itinerary as a PDF. This project was built to learn and demonstrate **LangChain's core capabilities** in a real, working product.

---



## 🧠 The LangChain Magic — What's Really Happening

This is the heart of the project. Here's exactly how LangChain powers Maya:

### 1. 🤖 LangChain Agent (`create_agent`)
Maya is not just an LLM responding to prompts. She is a **ReAct-style Agent** — she *thinks*, *decides which tool to call*, *observes the result*, and *thinks again* before replying.

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_message,
    checkpointer=st.session_state.memory,
)
```

This single block wires together the LLM, all tools, the system persona, and memory into one intelligent agent loop.

---

### 2. 🛠️ LangChain Tools (`Tool` + `TavilySearchResults`)
Tools are what make Maya *actually useful*. Each tool is a Python function wrapped so the agent can call it autonomously — Maya decides *when* and *which* tool to use based on the user's question.

```python
tools = [
    Tool(name="WeatherTool",       func=get_weather,        description="..."),
    Tool(name="AttractionsTool",   func=get_top_attractions, description="..."),
    Tool(name="LocationMapTool",   func=get_location_map,   description="..."),
    TavilySearchResults(max_results=3, include_images=True)
]
```

| Tool | What it does | API used |
|---|---|---|
| 🌤️ WeatherTool | Live weather for any city | OpenWeatherMap |
| 🏛️ AttractionsTool | Top 5 tourist spots | Geoapify Places |
| 🗺️ LocationMapTool | Geocodes places, renders map | Geoapify Geocoding |
| 🔍 TavilySearch | Web search for tips, food, budgets | Tavily AI |

The key LangChain concept here: the **tool's `description`** is what the agent reads to decide whether to call it. Write a bad description → agent won't use the tool correctly. This was a real learning moment. 💡

---

### 3. 🧠 Memory (`MemorySaver` + `langgraph`)
Maya remembers your entire conversation. If you say *"Plan a trip to Manali"* and then ask *"What should I pack?"* — she knows you're still talking about Manali. This is powered by LangGraph's `MemorySaver`:

```python
from langgraph.checkpoint.memory import MemorySaver

st.session_state.memory = MemorySaver()

config = {"configurable": {"thread_id": "maya_session"}}
agent.stream({"messages": [...]}, config=config)
```

The `thread_id` is what ties all messages together into one persistent conversation thread.

---

### 4. 📝 Prompt Template (System Prompt Engineering)
The system prompt is Maya's personality, instructions, and rules — all in one. This is where prompt engineering meets LangChain:

```python
system_message = (
    "You are Maya, a premium Indian travel concierge. "
    "You MUST create a highly detailed day-by-day itinerary. "
    "ALWAYS check weather, find attractions, and search the web. "
    "Display real image URLs from search results using HTML..."
)
```

Key lesson: telling the agent *when* to use tools (via the system prompt) vs *which* tool to use (via tool descriptions) is the real skill in agentic AI. 🎯

---

### 5. ⚡ Streaming (`agent.stream`)
Instead of waiting for the full response, Maya streams intermediate steps — you can see her thinking in real time:

```python
for event in agent.stream({"messages": [...]}, config=config, stream_mode="updates"):
    for node, data in event.items():
        if node == "tools":
            st.write(f"🔍 Used tool: {msg.name}")
```

This creates the "Maya is fetching live data..." experience that makes the app feel alive.

---

## 🌟 Features

- 🧠 **Intelligent Agent** — autonomously calls the right tools based on your question
- 🌤️ **Live Weather** — real-time conditions for your destination
- 🏛️ **Top Attractions** — actual tourist spots, not hallucinated ones
- 🗺️ **Interactive Map** — folium map with place markers rendered inline
- 🔍 **Web Search** — budget tips, food recommendations, travel hacks
- 💰 **Budget-Aware Planning** — slider adjusts recommendations (street food vs fine dining, budget vs luxury hotels)
- 🎒 **Smart Suggestion Chips** — one-click prompts for packing list, hotels, best season
- 📄 **PDF Export** — download your full itinerary as a PDF
- 🧠 **Conversation Memory** — remembers context across the entire chat session

---

## 🛠️ Tech Stack

| Tech | Role |
|---|---|
| 🦜 LangChain | Agent framework, tools, memory |
| 🟦 LangGraph | Memory checkpointing (`MemorySaver`) |
| 🤖 Groq (`llama-3.3-70b`) | The LLM powering Maya's brain |
| 🌐 Streamlit | Frontend UI |
| 🔍 Tavily | Web search tool |
| 🌤️ OpenWeatherMap | Live weather data |
| 📍 Geoapify | Places + geocoding |
| 🗺️ Folium | Interactive map rendering |
| 📄 fpdf2 | PDF export |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
https://github.com/Shrutij26/Travel-concierge-
cd maya-travel-concierge
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API keys
Create a `.env` file in the root folder:
```env
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweather_key
GEOAPIFY_API_KEY=your_geoapify_key
TAVILY_API_KEY=your_tavily_key
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔑 Getting Free API Keys

All of these are **free** with no credit card needed (except Groq, which has a generous free tier):

| API | Free Tier | Link |
|---|---|---|
| Groq | Free, fast LLM inference | [console.groq.com](https://console.groq.com) |
| OpenWeatherMap | 1M calls/month free | [openweathermap.org](https://openweathermap.org/api) |
| Geoapify | 3,000 requests/day free | [geoapify.com](https://www.geoapify.com) |
| Tavily | 1,000 searches/month free | [tavily.com](https://tavily.com) |

---

## 📁 Project Structure

```
maya-travel-concierge/
│
├── app.py              # Main application — all logic lives here
├── requirements.txt    # Python dependencies
├── .env                # API keys (DO NOT commit this!)
├── .gitignore          # Ignores .env and __pycache__
└── README.md           # You're reading this!
```

---

## 💡 Key LangChain Concepts I Learnt

Building this project taught me these real LangChain skills:

1. **How agents decide which tool to call** — it's all in the `description` field
2. **Why memory needs a `thread_id`** — to separate multiple user sessions
3. **The difference between a Chain and an Agent** — chains are fixed flows, agents are dynamic decision-makers
4. **How to inject user context into the agent** — by prepending info to the user message, not the system prompt
5. **How streaming works in LangGraph** — `stream_mode="updates"` gives node-level visibility into what the agent is doing

---



## ⭐ If you found this helpful, give it a star!

> *"The best way to learn LangChain is to build something real. Don't just follow tutorials — make something that actually does something."*
