# Local-Pokedex

Local AI-powered Pokédex using LangGraph + Ollama.

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1
python main.py
```

## Folder Structure

```
pokedex-agent/
├── main.py              # Entry point
├── agent.py             # LangGraph agent + graph
├── state.py             # PokedexState TypedDict
├── requirements.txt
├── tools/
│   ├── __init__.py
│   ├── pokeapi.py       # PokeAPI tool
│   ├── cache.py         # SQLite cache tool
├── data/
│   └── pokedex_cache.db # Cache of pokedex
└── ui/
    └── display.py       # Rich terminal UI helpers
```