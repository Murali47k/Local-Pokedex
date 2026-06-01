# Local-Pokedex

Local AI-powered Pokédex using **LangGraph + Ollama**.  
Two cooperating agents: a **Fetcher** that queries the PokéAPI and composes responses,  
and a **Verifier** that audits every answer before it reaches you.


## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1        # or any model you prefer
python main.py              # default: llama3.1
python main.py llama3.2     # specify a different model
python main.py llama3.1 --verbose   # show verifier verdict after each answer
```

## Runtime commands

| Input | Effect |
|---|---|
| Any Pokémon question | Fetcher fetches → Verifier audits → answer shown |
| `gen <number>` | Switch active generation (1–9) for move lookups |
| `verbose` | Toggle verifier output on/off |
| `exit` / `quit` / `q` | Quit |

## Folder structure

```
pokedex-agent/
├── main.py              # Entry point + streaming loop
├── agent.py             # LangGraph graph: Fetcher → Tools → Verifier
├── state.py             # PokedexState TypedDict (includes verified, verification_notes)
├── requirements.txt
├── tools/
│   ├── __init__.py
│   └── pokeapi.py       # get_pokemon_info, get_evolution_chain, get_moves, get_type_matchup
└── ui/
    ├── __init__.py
    └── display.py       # Rich terminal helpers
```

## How the dual-agent pipeline works

mermaid```
flowchart LR

    U[User]

    subgraph Fetch Phase
        F[Fetcher]
        T[Tool Node]
        F -->|Calls Tools| T
        T -->|Returns Data| F
    end

    subgraph Verification Phase
        V[Verifier]
    end

    R[Response]

    U --> F
    F -->|Draft Answer + Tool Results| V
    V -->|PASS / Corrected Output| R
```

The Verifier never calls tools itself — it only checks the Fetcher's output
against the raw API data that was already fetched.