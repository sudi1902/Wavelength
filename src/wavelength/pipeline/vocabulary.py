"""Label vocabulary for CLAP zero-shot classification.

Two prompt groups:
- SFX_LABELS: what an effect can be named. Short, CapCut/DAW-friendly words.
- OTHER_LABELS: separation bleed we want OUT of the library (speech, music,
  laughter, plain noise). When one of these outscores every SFX label, the
  segment goes to quarantine instead.

Each label maps to the text prompt CLAP scores against; multi-word prompts
give the text encoder more to grip than bare labels.
"""

PROMPT_TEMPLATE = "the sound of {}"

SFX_LABELS: dict[str, str] = {
    # movement / transitions
    "whoosh": "a whoosh passing by",
    "swoosh": "a fast swoosh swipe",
    "swipe": "a quick swipe transition",
    "riser": "a rising sweep building up",
    "rewind": "a tape rewinding",
    "tape-stop": "a tape stopping abruptly",
    "record-scratch": "a vinyl record scratch",
    # impacts
    "boom": "a deep boom",
    "explosion": "an explosion",
    "impact": "a heavy cinematic impact hit",
    "thud": "a dull thud",
    "punch": "a punch impact",
    "slap": "a slap",
    "slam": "a door slamming",
    "knock": "knocking on a door",
    "clap": "a single hand clap",
    "snap": "a finger snap",
    "stomp": "a foot stomping",
    # UI / notification
    "pop": "a small pop",
    "click": "a click",
    "tap": "a soft tap",
    "beep": "an electronic beep",
    "ding": "a ding",
    "chime": "a pleasant chime",
    "bell": "a bell ringing",
    "notification": "a phone notification alert",
    "camera-shutter": "a camera shutter",
    "keyboard": "typing on a keyboard",
    "cash-register": "a cash register cha-ching",
    "coin": "coins clinking",
    # tonal / musical stingers
    "sparkle": "a magical sparkle shimmer",
    "twinkle": "a twinkling glitter",
    "magic": "a magic spell chime",
    "harp": "a harp glissando",
    "sting": "a short dramatic musical sting",
    "airhorn": "an air horn blast",
    "horn": "a horn honking",
    "whistle": "a whistle",
    "slide-whistle": "a slide whistle",
    # cartoon
    "boing": "a cartoon boing spring",
    "squeak": "a squeak",
    "pow": "a cartoon pow hit",
    "zip": "a fast zip",
    # electronic / glitch
    "laser": "a laser zap",
    "zap": "an electric zap",
    "glitch": "a digital glitch",
    "buzz": "an electric buzz",
    "buzzer": "a wrong-answer buzzer",
    "alarm": "an alarm ringing",
    "static-burst": "a burst of static",
    "robot": "a robotic servo movement",
    # nature / physical
    "wind": "wind blowing",
    "splash": "a water splash",
    "drip": "a water drop dripping",
    "pour": "liquid pouring",
    "bubble": "bubbles bubbling",
    "fire": "fire crackling",
    "sizzle": "something sizzling in a pan",
    "thunder": "a thunder rumble",
    "glass-break": "glass shattering",
    "paper": "paper rustling",
    "page-turn": "a page turning",
    "footsteps": "footsteps walking",
    "door-creak": "a door creaking",
    "heartbeat": "a heartbeat thumping",
    "breath": "a sharp breath inhale",
    "kiss": "a kiss smack",
    "crunch": "a crunchy bite",
    "gulp": "a gulp swallowing",
    "animal": "an animal call",
    "bird": "a bird chirping",
    "dog-bark": "a dog barking",
    "engine": "an engine revving",
    "car-horn": "a car horn",
    "siren": "a siren wailing",
    "crowd": "a crowd cheering",
}

OTHER_LABELS: dict[str, str] = {
    "speech": "a person talking",
    "speech-2": "someone speaking words",
    "singing": "a person singing",
    "laughter": "people laughing",
    "scream": "a person screaming",
    "music": "music playing",
    "music-2": "a song with a melody and beat",
    "noise": "constant background noise",
    "room-tone": "silent room tone with faint hiss",
}


def build_prompts() -> list[dict]:
    """Flatten both groups into the worker's prompt spec."""
    prompts = []
    for label, text in SFX_LABELS.items():
        prompts.append(
            {"label": label, "group": "sfx", "text": PROMPT_TEMPLATE.format(text)}
        )
    for label, text in OTHER_LABELS.items():
        prompts.append(
            {"label": label.removesuffix("-2"), "group": "other",
             "text": PROMPT_TEMPLATE.format(text)}
        )
    return prompts
