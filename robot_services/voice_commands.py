"""Fixed vocabulary for simulation voice control."""

COMMANDS = {
    "walk forward": ("motion", "forward"),
    "walk backward": ("motion", "backward"),
    "turn left": ("motion", "turn_left"),
    "turn right": ("motion", "turn_right"),
    "stand up": ("posture", "Stand"),
    "sit down": ("posture", "Sit"),
    "stop": ("stop", None),
}


def resolve_command(phrase):
    """Return the exact normalized command, or None for an unrecognized phrase."""
    if not isinstance(phrase, str):
        return None
    return COMMANDS.get(" ".join(phrase.casefold().split()))
