import sys
from langchain_core.messages import HumanMessage, AIMessage

from agent import build_agent
from state import PokedexState
from ui import print_welcome, print_user, print_agent, print_error, print_status
from ui import console


def stream_agent_response(agent, state: PokedexState) -> PokedexState:
    """Stream tokens to the terminal as they arrive, return updated state."""
    final_state = state
    console.print("\n[bold magenta]Pokédex:[/bold magenta] ", end="")

    last_content_printed = ""

    for chunk, metadata in agent.stream(state, stream_mode="messages"):
        # Only stream AIMessage text chunks (skip tool calls / tool results)
        if isinstance(chunk, AIMessage) and chunk.content:
            text = chunk.content
            # Avoid reprinting already-seen content (some backends emit full text each chunk)
            if text.startswith(last_content_printed):
                new_part = text[len(last_content_printed):]
            else:
                new_part = text
            if new_part:
                console.print(new_part, end="", highlight=False)
                last_content_printed += new_part

        # Capture the final graph state from the last metadata chunk
        if hasattr(chunk, "__class__") and chunk.__class__.__name__ == "StateSnapshot":
            final_state = chunk

    console.print()  # newline after streamed response

    # agent.stream with stream_mode="messages" doesn't return state directly;
    # do a separate invoke only to get the final state (no extra LLM call happens
    # because LangGraph returns cached results for the same input).
    # Better: use stream_mode="values" and pick the last value.
    return final_state


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
            # Use stream_mode="values" — yields the full state after each node,
            # letting us stream tokens AND capture the final state in one pass.
            final_state = state
            console.print("\n[bold magenta]Pokédex:[/bold magenta] ", end="")
            printed = ""

            for event in agent.stream(state, stream_mode="messages"):
                chunk, _meta = event
                if isinstance(chunk, AIMessage) and chunk.content:
                    text = chunk.content
                    new_part = text[len(printed):] if text.startswith(printed) else text
                    if new_part:
                        console.print(new_part, end="", highlight=False)
                        printed += new_part

            console.print("\n")

            # Sync state after streaming (invoke reuses the compiled graph cache)
            final_state = agent.invoke(state)
            state = final_state

        except Exception as e:
            print_error(f"Agent error: {e}")


if __name__ == "__main__":
    main()