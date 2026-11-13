# 🔎 Multi-Agent AI Research System

## Introduction
A multi-agent AI system that researches any topic end to end. It searches the web, reads the best source in depth, writes a structured report, and then critiques its own work. It runs from the terminal or through a Streamlit web UI.

## Working
The system chains four stages:

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | Search Agent | Finds recent, reliable sources (title, URL, snippet) |
| 2 | Reader Agent | Picks the most relevant URL and scrapes it for deeper content |
| 3 | Writer Chain | Combines search and scraped content into a research report |
| 4 | Critic Chain | Reviews the report and returns feedback |

Topic ─► Search Agent ─► Reader Agent ─► Writer Chain ─► Critic Chain ─► Report + Feedback

## Project Structure
```
Multi-Agent-AI-Research-System/
├── agents.py          # Search/reader agents, writer and critic chains
├── tools.py           # Tools used by the agents (web search, web scraper)
├── callerFunc.py      # Runs the full pipeline in the terminal
├── app.py             # Streamlit web UI
├── requirements.txt   # Python dependencies
├── .env               # API keys (not committed)
└── README.md
```

## Tech Stack
- Python
- LangChain (agents and chains)
- Streamlit (web interface)
- python-dotenv (environment variables)
