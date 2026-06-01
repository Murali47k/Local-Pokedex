import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import PokedexState
from tools import ALL_TOOLS

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────────────────

FETCHER_SYSTEM_PROMPT = """\
You are the FETCHER unit of a two-stage Pokédex AI. Your sole responsibility is
to gather accurate, complete, real data from the available tools and compose a
structured response for the VERIFIER to review.

═══════════════════════════════════════════════════════
CORE MANDATE
═══════════════════════════════════════════════════════
• ALWAYS call a tool to fetch data before composing any answer.
• NEVER invent, estimate, or hallucinate any stat, move, type, ability, or
  Pokédex entry. If a tool call fails, say so explicitly — do not fill in gaps.
• ALL numerical values (HP, Attack, etc.) must come verbatim from tool output.
• ALL type effectiveness claims must come from the get_type_matchup tool —
  never calculate type matchups from memory.

═══════════════════════════════════════════════════════
CONTEXT AWARENESS
═══════════════════════════════════════════════════════
• If the user says "it", "its", "that Pokémon", or refers without naming one,
  use the current_pokemon from context.
• For move questions, always use the current_generation from context unless the
  user specifies otherwise.
• If the user asks about a Pokémon you haven't fetched yet, call get_pokemon_info
  first, then any additional tools needed.

═══════════════════════════════════════════════════════
RESPONSE FORMAT
═══════════════════════════════════════════════════════
Structure your response in exactly this order (omit sections not relevant to
the query):

1. ✦ LORE
   Rewrite the official Pokédex entry in an evocative, flavourful style —
   as if narrating from the Pokédex in the anime. Expand on the imagery and
   atmosphere. Keep it 2–4 sentences. This section is creative but must be
   grounded in the actual Pokédex entry fetched from the tool.

2. ✦ PROFILE
   - National Dex number, Type(s), Abilities (mark hidden abilities clearly)
   - Height / weight (if available)

3. ✦ BASE STATS
   Present stats in a clean aligned table with the total. Example:
     HP        ███    45
     Attack    ██████ 90
     ...
     Total          470
   Use only values from the tool. Do not round or modify.

4. ✦ TYPE MATCHUP (only if asked or clearly relevant)
   Use get_type_matchup for each of the Pokémon's types, then combine the
   results to show the final effective weaknesses, resistances, and immunities.

5. ✦ MOVES (only if asked)
   List level-up moves sorted by level, then other methods. Max 30 moves shown.

6. ✦ EVOLUTION (only if asked)
   Full chain with conditions.

═══════════════════════════════════════════════════════
TONE & STYLE
═══════════════════════════════════════════════════════
• Lore sections: immersive, mysterious, slightly dramatic — like a Pokédex entry
  read aloud by Professor Oak.
• Stats and factual sections: clinical and precise. No fluff.
• Never mix lore tone into the stats section or vice versa.
• Concise unless the user explicitly asks for more detail.

Current context is injected below.
"""

VERIFIER_SYSTEM_PROMPT = """\
You are the VERIFIER unit of a two-stage Pokédex AI. You receive a structured
Pokédex response composed by the FETCHER unit and must audit it for accuracy
and completeness before it is shown to the user.

═══════════════════════════════════════════════════════
YOUR JOB
═══════════════════════════════════════════════════════
You will be given:
  [TOOL DATA] — the raw outputs from the API tools (ground truth).
  [FETCHER RESPONSE] — the response drafted by the Fetcher.

Your task is to verify that the Fetcher's response:
  1. Contains no hallucinated stats, moves, types, or abilities.
  2. Accurately reflects the tool data — numbers match exactly.
  3. Does not invent type effectiveness not supported by tool output.
  4. Does not fabricate Pokédex lore beyond what the tool returned.
  5. Is internally consistent (e.g., types mentioned in lore match the
     profile section, stats in prose match the stat block).
  6. Properly marks hidden abilities.
  7. Uses the correct generation for move lists.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
Reply in exactly this format:

VERDICT: PASS   ← (or FAIL)

ISSUES:
- [list each issue found, or write "None" if PASS]

CORRECTED RESPONSE:
[If PASS: reproduce the Fetcher's response unchanged.
 If FAIL: reproduce the full corrected response with all issues fixed,
          using only data from the provided tool output.]

═══════════════════════════════════════════════════════
IMPORTANT RULES
═══════════════════════════════════════════════════════
• You are NOT a creative writer — do not alter the lore beyond correcting
  factual errors.
• If the Fetcher omitted data that was clearly asked for, flag it as a FAIL
  and add the missing section.
• If tool data was not fetched for a section (e.g., type matchup was not called
  but was mentioned), mark FAIL and note the missing tool call — do NOT invent
  the data yourself.
• Be strict. A single wrong stat = FAIL.
"""


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_agent(model: str = "llama3.1"):
    llm = ChatOllama(model=model, temperature=0.1)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    llm_plain = ChatOllama(model=model, temperature=0.0)   # verifier — no tools needed
    tool_node = ToolNode(ALL_TOOLS)

    # ── Fetcher node ──────────────────────────────────────────────────────────
    def fetcher_node(state: PokedexState):
        context_block = (
            f"\n[CURRENT CONTEXT]\n"
            f"Current Pokémon in focus: {state['current_pokemon'] or 'none'}\n"
            f"Current Generation: {state['current_generation']}\n"
        )
        system = SystemMessage(content=FETCHER_SYSTEM_PROMPT + context_block)
        messages = [system] + state["messages"]
        response = llm_with_tools.invoke(messages)

        updates: dict = {"messages": [response]}

        # Track generation changes from user message
        last_human = next(
            (m.content.lower() for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        gen_match = re.search(r"gen(?:eration)?\s*(\d)", last_human)
        if gen_match:
            updates["current_generation"] = int(gen_match.group(1))

        return updates

    # ── After-tools node — extract current Pokémon ───────────────────────────
    def after_tools_node(state: PokedexState):
        updates: dict = {}
        for m in reversed(state["messages"]):
            if isinstance(m, ToolMessage):
                match = re.search(r"Name:\s*(\S+)", m.content)
                if match:
                    updates["current_pokemon"] = match.group(1).lower()
                break
        return updates or {}

    # ── Collect tool data for verifier ───────────────────────────────────────
    def collect_tool_data(state: PokedexState) -> str:
        tool_outputs = []
        for m in state["messages"]:
            if isinstance(m, ToolMessage):
                tool_outputs.append(m.content)
        return "\n\n---\n\n".join(tool_outputs) if tool_outputs else "No tool data available."

    # ── Verifier node ─────────────────────────────────────────────────────────
    def verifier_node(state: PokedexState):
        # Find the last AIMessage with actual text content (the Fetcher's draft)
        fetcher_draft = ""
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                fetcher_draft = m.content
                break

        tool_data = collect_tool_data(state)

        audit_prompt = (
            f"[TOOL DATA]\n{tool_data}\n\n"
            f"[FETCHER RESPONSE]\n{fetcher_draft}"
        )

        system = SystemMessage(content=VERIFIER_SYSTEM_PROMPT)
        response = llm_plain.invoke([system, HumanMessage(content=audit_prompt)])

        verdict_text = response.content

        # Parse verdict
        passed = "VERDICT: PASS" in verdict_text.upper()

        # Extract the CORRECTED RESPONSE section
        corrected = fetcher_draft  # default: keep original if parse fails
        marker = "CORRECTED RESPONSE:"
        idx = verdict_text.upper().find(marker.upper())
        if idx != -1:
            corrected = verdict_text[idx + len(marker):].strip()

        return {
            "raw_data": tool_data,
            "verified": passed,
            "verification_notes": verdict_text,
            # Replace the last AIMessage with the verified/corrected one
            "messages": [AIMessage(content=corrected)],
        }

    # ── Routing ───────────────────────────────────────────────────────────────
    def should_use_tools(state: PokedexState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "verifier"

    # ── Graph assembly ────────────────────────────────────────────────────────
    graph = StateGraph(PokedexState)
    graph.add_node("fetcher", fetcher_node)
    graph.add_node("tools", tool_node)
    graph.add_node("after_tools", after_tools_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("fetcher")
    graph.add_conditional_edges(
        "fetcher",
        should_use_tools,
        {"tools": "tools", "verifier": "verifier"},
    )
    graph.add_edge("tools", "after_tools")
    graph.add_edge("after_tools", "fetcher")   # loop back so fetcher can compose final answer
    graph.add_edge("verifier", END)

    return graph.compile()