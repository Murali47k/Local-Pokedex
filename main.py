import sys
from langchain_core.messages import HumanMessage

from agent import build_agent
from state import PokedexState
from ui import print_welcome,print_user,print_agent,print_error,print_status


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "llama3.1"
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

        # Allow setting generation directly
        if user_input.lower().startswith("gen "):
            try:
                gen = int(user_input.split()[1])
                state["current_generation"] = gen
                print_agent(f"Generation set to {gen}.")
                continue
            except ValueError:
                pass

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        try:
            result = agent.invoke(state)
            state = result

            last_ai = None
            for m in reversed(state["messages"]):
                from langchain_core.messages import AIMessage
                if isinstance(m, AIMessage) and m.content:
                    last_ai = m.content
                    break

            if last_ai:
                print_agent(last_ai)
            else:
                print_agent("(No response — tool called but no follow-up text.)")

        except Exception as e:
            print_error(f"Agent error: {e}")


if __name__ == "__main__":
    main()