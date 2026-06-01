from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class PokedexState(TypedDict):
    """Represents the state of the Pokédex conversation."""
    messages: Annotated[list, add_messages]
    current_pokemon: str
    current_generation: int
    raw_data: Optional[str]          # Fetcher agent's raw output before verification
    verified: Optional[bool]         # Did the verifier approve the data?
    verification_notes: Optional[str] # Verifier's notes / corrections