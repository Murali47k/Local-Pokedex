import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import PokedexState
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Pokédex AI — precise, factual, and tool-driven.

STRICT RULES (never break these):
1. ALWAYS call a tool before stating any base stat, type, ability, move, or evolution.
   Never recall these from memory — they change between games and you will be wrong.
2. Type matchups (e.g. "super effective", "immune") must be derived from the fetched type data only.
   Do not calculate or guess type interactions yourself.
3. If the user refers to "it", "that one", "its", or "the previous one" — they mean the last Pokémon
   discussed. Use current_pokemon from context. Do not ask for clarification, just use it.
4. Generation matters for moves. Always use current_generation from context unless the user specifies.
   A move legal in Gen 9 may not exist in Gen 4 — never assume availability across gens.
5. Never invent Pokédex flavour text. If you write a dex entry, clearly label it as a summary,
   not an official game quote.

RESPONSE STYLE:
- Lead with the most relevant info. No filler like "Great question!" or "Sure!".
- Stats and moves in a clean list. Prose for lore/dex entries.
- Keep it short unless the user asks for detail.
- If a tool call fails, say so honestly. Do not fill in from memory.

Current context will be injected automatically."""


def build_agent(model: str = "llama3.1"):
    llm = ChatOllama(model=model, temperature=0.1)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def should_continue(state: PokedexState):
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    def call_model(state: PokedexState):
        context = (
            f"\nCurrent Pokémon: {state['current_pokemon'] or 'none'}"
            f"\nCurrent Generation: {state['current_generation']}"
        )
        system = SystemMessage(content=SYSTEM_PROMPT + context)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)

        updates: dict = {"messages": [response]}

        # Update context inline — no separate node needed
        last_human = next(
            (m.content.lower() for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        gen_match = re.search(r"gen(?:eration)?\s*(\d)", last_human)
        if gen_match:
            updates["current_generation"] = int(gen_match.group(1))

        return updates

    def after_tools(state: PokedexState):
        """Extract current_pokemon from the latest tool result."""
        updates: dict = {}
        for m in reversed(state["messages"]):
            if isinstance(m, ToolMessage):
                match = re.search(r"Name:\s*(\w+)", m.content)
                if match:
                    updates["current_pokemon"] = match.group(1).lower()
                break
        return updates if updates else {}

    graph = StateGraph(PokedexState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("after_tools", after_tools)

    graph.set_entry_point("agent")                        # removed redundant context node
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "after_tools")
    graph.add_edge("after_tools", "agent")

    return graph.compile()