from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class PokedexState(TypedDict):
    messages: Annotated[list, add_messages]
    current_pokemon: str
    current_generation: int