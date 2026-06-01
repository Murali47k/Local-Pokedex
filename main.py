import sys
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent
from state import PokedexState
from ui import console, print_welcome, print_user, print_agent, print_verifier, print_error, print_status


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1"
    verbose = "--verbose" in sys.argv   # show verifier output

    print_welcome()
    print_status(f"Loading model: {model}")

    try:
        agent = build_agent(model)
    except Exception as e:
        print_error(f"Failed to load agent: {e}")
        print_error("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    state: PokedexState = {
        "messages": [],
        "current_pokemon": "",
        "current_generation": 9,
        "raw_data": None,
        "verified": None,
        "verification_notes": None,
    }

    print_status("Ready")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "verbose":
            verbose = not verbose
            print_status(f"Verifier output {'ON' if verbose else 'OFF'}")
            continue

        if user_input.lower().startswith("gen "):
            try:
                gen = int(user_input.split()[1])
                state["current_generation"] = gen
                print_agent(f"Generation set to {gen}.")
                continue
            except ValueError:
                print_error("Usage: gen <number>  e.g. gen 4")
                continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        try:
            # ── Stream tokens from the fetcher as they arrive ─────────────────
            console.print("\n[bold magenta]Pokédex:[/bold magenta] ", end="")
            printed = ""

            final_state = state

            for event in agent.stream(state, stream_mode="messages"):
                chunk, meta = event
                node = meta.get("langgraph_node", "")

                # Stream only the fetcher's final text output (skip tool calls)
                if (
                    node == "fetcher"
                    and isinstance(chunk, AIMessage)
                    and chunk.content
                    and not getattr(chunk, "tool_calls", None)
                ):
                    text = chunk.content
                    new_part = text[len(printed):] if text.startswith(printed) else text
                    if new_part:
                        console.print(new_part, end="", highlight=False)
                        printed += new_part

            console.print("\n")

            # ── Capture full final state ──────────────────────────────────────
            final_state = agent.invoke(state)

            # ── Show verifier notes if verbose ────────────────────────────────
            if verbose and final_state.get("verification_notes"):
                notes = final_state["verification_notes"]
                verdict_line = next(
                    (l for l in notes.splitlines() if "VERDICT" in l.upper()), ""
                )
                print_verifier(verdict_line or notes[:200])

            state = final_state

        except Exception as e:
            print_error(f"Agent error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()