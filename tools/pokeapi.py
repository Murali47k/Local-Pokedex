import requests
from langchain_core.tools import tool

BASE_URL = "https://pokeapi.co/api/v2"

_session = requests.Session()
_session.headers.update({"User-Agent": "local-pokedex/1.0"})


def _fetch(endpoint: str) -> dict:
    r = _session.get(f"{BASE_URL}/{endpoint}", timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def get_pokemon_info(name: str) -> str:
    """
    Fetch base stats, types, abilities, and species flavour text for a Pokémon.
    Returns structured text that includes the official Pokédex entry (lore),
    base stats, type(s), and ability list.
    """
    name = name.lower().strip()

    try:
        data = _fetch(f"pokemon/{name}")
    except Exception as e:
        return f"Error fetching Pokémon '{name}': {e}"

    types = "/".join(t["type"]["name"].capitalize() for t in data["types"])
    abilities = ", ".join(
        a["ability"]["name"].replace("-", " ").title()
        + (" (Hidden)" if a["is_hidden"] else "")
        for a in data["abilities"]
    )
    stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
    total = sum(stats.values())

    # Fetch species for lore (Pokédex flavour text)
    dex_entry = "No Pokédex entry found."
    try:
        species = _fetch(f"pokemon-species/{name}")
        # Prefer the most recent English entry
        english_entries = [
            e for e in species.get("flavor_text_entries", [])
            if e["language"]["name"] == "en"
        ]
        if english_entries:
            raw_text = english_entries[-1]["flavor_text"]
            # PokeAPI embeds form-feed / newline control chars — clean them
            dex_entry = raw_text.replace("\f", " ").replace("\n", " ").strip()
    except Exception:
        pass

    result = (
        f"Name: {data['name'].capitalize()}\n"
        f"National Dex #: {data['id']}\n"
        f"Type: {types}\n"
        f"Abilities: {abilities}\n"
        f"Pokédex Entry: {dex_entry}\n"
        f"--- Base Stats ---\n"
        f"HP:       {stats.get('hp', '?')}\n"
        f"Attack:   {stats.get('attack', '?')}\n"
        f"Defense:  {stats.get('defense', '?')}\n"
        f"Sp. Atk:  {stats.get('special-attack', '?')}\n"
        f"Sp. Def:  {stats.get('special-defense', '?')}\n"
        f"Speed:    {stats.get('speed', '?')}\n"
        f"Total:    {total}\n"
    )
    return result


@tool
def get_evolution_chain(name: str) -> str:
    """
    Fetch the full evolution chain for a Pokémon, including trigger conditions
    (level-up threshold, held item, trade, friendship, etc.).
    """
    name = name.lower().strip()

    try:
        species = _fetch(f"pokemon-species/{name}")
        chain_url = species["evolution_chain"]["url"]
        chain_id = chain_url.rstrip("/").split("/")[-1]
        chain_data = _fetch(f"evolution-chain/{chain_id}")
    except Exception as e:
        return f"Error fetching evolution chain for '{name}': {e}"

    def _condition(detail: dict) -> str:
        parts = []
        trigger = detail.get("trigger", {}).get("name", "")
        level = detail.get("min_level")
        item = detail.get("item")
        held_item = detail.get("held_item")
        happiness = detail.get("min_happiness")
        time_of_day = detail.get("time_of_day")
        known_move = detail.get("known_move")
        location = detail.get("location")
        gender = detail.get("gender")

        if level:
            parts.append(f"Lv. {level}")
        if item:
            parts.append(f"use {item['name'].replace('-', ' ').title()}")
        if held_item:
            parts.append(f"hold {held_item['name'].replace('-', ' ').title()}")
        if happiness:
            parts.append(f"happiness ≥ {happiness}")
        if time_of_day:
            parts.append(f"{time_of_day}")
        if known_move:
            parts.append(f"know {known_move['name'].replace('-', ' ').title()}")
        if location:
            parts.append(f"at {location['name'].replace('-', ' ').title()}")
        if gender is not None:
            parts.append("♀" if gender == 1 else "♂")
        if trigger == "trade" and not parts:
            parts.append("Trade")
        return f"({', '.join(parts)})" if parts else f"({trigger})"

    def _parse(node, indent=0) -> list[str]:
        lines = []
        prefix = "  " * indent
        arrow = "→ " if indent > 0 else ""
        species_name = node["species"]["name"].capitalize()

        evo_details = node.get("evolution_details", [])
        if evo_details:
            cond = _condition(evo_details[0])
            lines.append(f"{prefix}{arrow}{species_name} {cond}")
        else:
            lines.append(f"{prefix}{arrow}{species_name}")

        for child in node.get("evolves_to", []):
            lines.extend(_parse(child, indent + 1))
        return lines

    chain_lines = _parse(chain_data["chain"])
    return "Evolution Chain:\n" + "\n".join(chain_lines)


@tool
def get_moves(name: str, generation: int = 9) -> str:
    """
    Fetch the moves a Pokémon can learn in a given generation (default: 9).
    Returns level-up moves sorted by level, followed by TM/HM/tutor/egg moves.
    """
    name = name.lower().strip()

    gen_map = {
        1: "red-blue",      2: "gold-silver",   3: "ruby-sapphire",
        4: "diamond-pearl", 5: "black-white",    6: "x-y",
        7: "sun-moon",      8: "sword-shield",   9: "scarlet-violet",
    }
    version_group = gen_map.get(generation, "scarlet-violet")

    try:
        data = _fetch(f"pokemon/{name}")
    except Exception as e:
        return f"Error fetching moves for '{name}': {e}"

    level_up, other = [], []
    for m in data["moves"]:
        for vg in m["version_group_details"]:
            if vg["version_group"]["name"] == version_group:
                method = vg["move_learn_method"]["name"]
                level = vg.get("level_learned_at", 0)
                move_name = m["move"]["name"].replace("-", " ").title()
                if method == "level-up":
                    level_up.append((level, move_name))
                else:
                    other.append((method.replace("-", " ").title(), move_name))

    if not level_up and not other:
        return f"{name.capitalize()} has no recorded moves for Generation {generation}."

    level_up.sort()
    other.sort()

    lines = [f"Moves for {name.capitalize()} (Gen {generation} — {version_group}):"]
    if level_up:
        lines.append("\nLevel-Up Moves:")
        for lvl, mv in level_up[:30]:
            lines.append(f"  Lv.{lvl:>3}  {mv}")
    if other:
        lines.append("\nOther Moves (TM / Egg / Tutor):")
        for method, mv in other[:30]:
            lines.append(f"  [{method}]  {mv}")

    return "\n".join(lines)


@tool
def get_type_matchup(type_name: str) -> str:
    """
    Fetch offensive and defensive type matchups for a given type.
    Returns what the type is super-effective, not-very-effective, or immune against
    both offensively and defensively.
    """
    type_name = type_name.lower().strip()
    try:
        data = _fetch(f"type/{type_name}")
    except Exception as e:
        return f"Error fetching type '{type_name}': {e}"

    dr = data["damage_relations"]

    def names(lst): return ", ".join(t["name"].capitalize() for t in lst) or "None"

    return (
        f"Type: {type_name.capitalize()}\n"
        f"\n[Offensive — this type's moves hit:]"
        f"\n  Super-effective vs:  {names(dr['double_damage_to'])}"
        f"\n  Not very effective vs: {names(dr['half_damage_to'])}"
        f"\n  No effect vs:        {names(dr['no_damage_to'])}"
        f"\n\n[Defensive — moves against this type:]"
        f"\n  Weak to:    {names(dr['double_damage_from'])}"
        f"\n  Resists:    {names(dr['half_damage_from'])}"
        f"\n  Immune to:  {names(dr['no_damage_from'])}"
    )