import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import PokedexState
from tools import ALL_TOOLS

SYSTEM_PROMPT = """
You are a Pokédex AI assistant.

Rules:

1. Always use tools when Pokémon data is needed.
2. Never hallucinate stats, moves, abilities, evolutions, learnsets, or Pokédex entries.
3. If a tool fails, report the error instead of guessing.
4. If the user says "it", "its", or "that Pokémon", use current_pokemon.
5. For move questions, use current_generation unless specified. Default is Generation 9.
6. Answer only what the user asked.
7. Do not add extra sections or information.
8. Do not explain your reasoning.
9. Keep answers concise unless the user asks for detail.

Formatting:

* Stats → clean stats like Aligned stat block with bar and total. Example:

HP      ████░░ 100

Attack  ████░░ 100

Defense ████░░ 100

Sp. Atk ████░░ 100

Sp. Def ████░░ 100

Speed   ████░░ 100

Total  600

Use only values from the tool. Do not round or modify..
* Moves → move list.
* Evolution → evolution chain.
* Type matchup → weaknesses, resistances, immunities.
* General Pokémon overview → short Pokédex entry + relevant facts.

Never say things like:
"Since you asked only for..."
"I have shown only..."
"Let me know if you need more."

Just answer directly.

Current context is injected below.
"""



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