import requests
from langchain_core.tools import tool
from .cache import get_cached, set_cached


BASE_URL = "https://pokeapi.co/api/v2"


def _fetch(endpoint: str) -> dict:
    r = requests.get(f"{BASE_URL}/{endpoint}", timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def get_pokemon_info(name: str) -> str:
    """Get base stats, types, and abilities for a Pokémon by name."""
    name = name.lower().strip()

    cached = get_cached(name)
    if cached:
        return cached

    try:
        data = _fetch(f"pokemon/{name}")
    except Exception as e:
        return f"Error fetching {name}: {e}"

    types = "/".join(t["type"]["name"].capitalize() for t in data["types"])
    abilities = ", ".join(
        a["ability"]["name"].replace("-", " ").title()
        + (" (hidden)" if a["is_hidden"] else "")
        for a in data["abilities"]
    )
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

    result = (
        f"Name: {data['name'].capitalize()}\n"
        f"Type: {types}\n"
        f"Abilities: {abilities}\n"
        f"HP: {stats.get('hp')}\n"
        f"Attack: {stats.get('attack')}\n"
        f"Defense: {stats.get('defense')}\n"
        f"Sp. Atk: {stats.get('special-attack')}\n"
        f"Sp. Def: {stats.get('special-defense')}\n"
        f"Speed: {stats.get('speed')}\n"
    )

    set_cached(name, result)
    return result


@tool
def get_evolution_chain(name: str) -> str:
    """Get the full evolution chain for a Pokémon."""
    name = name.lower().strip()
    try:
        species = _fetch(f"pokemon-species/{name}")
        chain_url = species["evolution_chain"]["url"]
        chain_id = chain_url.rstrip("/").split("/")[-1]
        chain_data = _fetch(f"evolution-chain/{chain_id}")
    except Exception as e:
        return f"Error fetching evolution chain for {name}: {e}"

    def parse_chain(node):
        lines = []
        current = node["species"]["name"].capitalize()
        for detail in node.get("evolution_details", []):
            trigger = detail.get("trigger", {}).get("name", "")
            level = detail.get("min_level")
            item = detail.get("item")
            if level:
                current = f"{current} (Level {level})"
            elif item:
                current = f"{current} (use {item['name']})"
            elif trigger == "trade":
                current = f"{current} (Trade)"
        lines.append(current)
        for evo in node.get("evolves_to", []):
            lines.append("  -> " + "\n  -> ".join(parse_chain(evo)))
        return lines

    chain = chain_data["chain"]
    return "\n".join(parse_chain(chain))


@tool
def get_moves(name: str, generation: int = 9) -> str:
    """Get moves a Pokémon can learn in a given generation."""
    name = name.lower().strip()
    gen_map = {
        1: "red-blue", 2: "gold-silver", 3: "ruby-sapphire",
        4: "diamond-pearl", 5: "black-white", 6: "x-y",
        7: "sun-moon", 8: "sword-shield", 9: "scarlet-violet",
    }
    version_group = gen_map.get(generation, "scarlet-violet")

    try:
        data = _fetch(f"pokemon/{name}")
    except Exception as e:
        return f"Error: {e}"

    moves = []
    for m in data["moves"]:
        for vg in m["version_group_details"]:
            if vg["version_group"]["name"] == version_group:
                method = vg["move_learn_method"]["name"]
                level = vg.get("level_learned_at", 0)
                move_name = m["move"]["name"].replace("-", " ").title()
                if method == "level-up":
                    moves.append(f"Lv.{level:>3}  {move_name}")
                else:
                    moves.append(f"  [{method}]  {move_name}")

    if not moves:
        return f"{name.capitalize()} has no move data for Generation {generation}."

    moves.sort()
    return f"Moves for {name.capitalize()} (Gen {generation}):\n" + "\n".join(moves[:40])