from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import PokedexState
from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Pokédex AI assistant — knowledgeable, concise, and factual.

Rules:
1. Always use tools to fetch real data. Never hallucinate stats, moves, or learnsets.
2. Remember context: if the user says "it" or "its", they mean the last Pokémon discussed.
3. Track the current generation for move questions. Default is Generation 9.
4. Give concise answers unless the user asks for detail.
5. Format stats and move lists clearly.

Current context will be injected into the conversation automatically."""


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
        return {"messages": [response]}

    def update_context(state: PokedexState):
        """Extract current_pokemon and generation from the conversation."""
        updates = {}
        last_human = None
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                last_human = m.content.lower()
                break

        if last_human:
            # Simple generation extraction: "gen 4", "generation 4"
            import re
            gen_match = re.search(r"gen(?:eration)?\s*(\d)", last_human)
            if gen_match:
                updates["current_generation"] = int(gen_match.group(1))

        # Extract Pokémon name from the last tool call result if any
        for m in reversed(state["messages"]):
            if isinstance(m, ToolMessage):
                import re
                match = re.search(r"Name:\s*(\w+)", m.content)
                if match:
                    updates["current_pokemon"] = match.group(1).lower()
                break

        return updates if updates else {}

    graph = StateGraph(PokedexState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_node("context", update_context)

    graph.set_entry_point("context")
    graph.add_edge("context", "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")

    return graph.compile()