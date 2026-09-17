FILLER = [
    "The committee met on Tuesday to review the quarterly figures.",
    "Rainfall in the northern districts stayed below the seasonal average.",
    "A new footbridge opened near the old railway yard last spring.",
    "The library extended its opening hours during the examination period.",
    "Several varieties of wheat were planted in the eastern fields.",
    "The museum acquired a collection of maps from a private donor.",
    "Traffic on the coastal road was diverted for three weeks.",
    "Researchers presented their findings at the annual meeting.",
    "The cafe on the corner changed owners in the autumn.",
    "Snow closed the mountain pass earlier than usual this year.",
    "The ferry timetable changes at the end of the month.",
    "A local choir rehearses in the hall on Thursday evenings.",
]
WINGS  = [f"the {d} {n}" for d in ("west", "north", "east", "south")
          for n in ("wing", "annexe", "gallery", "stairwell", "workshop", "store")]
ROOMS  = [f"room {n}{c}" for n in range(10, 40) for c in "ABCD"]
PLACES = WINGS + ROOMS

FORMS = [
    "The access code for {k} is {v}.",
    "{k} uses credential {v}.",
    "Entry to {k} requires the sequence {v}.",
    "Security identifier for {k}: {v}.",
    "Staff enter {v} to unlock {k}.",
    "{k} was reassigned the code {v} last quarter.",
]
ASK_PLAIN  = "\n\nQuestion: What is the access code for {k}?\nAnswer: The access code for {k} is"
ASK_PARA   = "\n\nQuestion: Which credential should be entered to open {k}?\nAnswer:"
ASK_LATEST = "\n\nQuestion: What is the current access code for {k}, after any change?\nAnswer:"
ASK_COUNT  = "\n\nQuestion: How many access codes in the text above begin with the digit {d}?\nAnswer:"


def make_code(rng, first=None):
    head = str(rng.randint(1000, 9999)) if first is None else str(first) + str(rng.randint(100, 999))
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTVWXYZ") for _ in range(2))
    return f"{head}-{letters}{rng.randint(0, 9)}"


_SIZES = {}


def sentence_sizes(tok):
    # Keyed by name: id() can be reused once a tokenizer is freed.
    if tok.name_or_path not in _SIZES:
        filler = sum(len(tok(s + " ", add_special_tokens=False).input_ids) for s in FILLER) / len(FILLER)
        fact = len(tok(FORMS[0].format(k="room 12A", v="1234-AB5") + " ", add_special_tokens=False).input_ids)
        _SIZES[tok.name_or_path] = (filler, fact)
    return _SIZES[tok.name_or_path]


def build_case(tok, probe, target_tokens, depth, rng):
    # Returns (prompt, gold, tokens). The prompt never exceeds target_tokens.
    if probe == "nearkey":
        # Four keys one character apart. v3 drew the other distractors from the same
        # pool, so the target key reappeared with a different code in 53% of prompts.
        stem = rng.randint(10, 39)
        near = [f"room {stem}{c}" for c in "ABCD"]
        names = near + rng.sample([r for r in ROOMS if r not in near], DISTRACTORS)
    else:
        names = rng.sample(PLACES, DISTRACTORS + 3)
    target, code = names[0], make_code(rng)
    plain = probe in ("simple", "multikey", "nearkey")
    form = (lambda: FORMS[0] if plain else rng.choice(FORMS))

    facts, ask, gold, revision = [], ASK_PLAIN.format(k=target), " " + code, None
    if probe == "twohop":
        facts = [FORMS[0].format(k=names[1], v=code),
                 f"{target} shares the same access code as {names[1]}."]
    elif probe == "update":
        facts = [form().format(k=target, v=make_code(rng))]      # the stale value
        revision = f"The access code for {target} was changed to {code}."
        ask = ASK_LATEST.format(k=target)
    elif probe != "count":
        facts = [form().format(k=target, v=code)]
        ask = (ASK_PLAIN if plain else ASK_PARA).format(k=target)

    competitors = []
    if probe != "simple":
        others = names[2:]
        digit = rng.choice("2345678")
        # v3 marked every fifth code, so the count was always 13; now it varies.
        marked = set(rng.sample(range(len(others)), rng.randint(6, 20))) if probe == "count" else set()
        for i, other in enumerate(others):
            first = (digit if i in marked else rng.choice("19")) if probe == "count" else None
            competitors.append(form().format(k=other, v=make_code(rng, first)))
        if probe == "count":
            ask, gold = ASK_COUNT.format(d=digit), " " + str(len(marked))

    filler_len, fact_len = sentence_sizes(tok)
    fixed = len(competitors) + len(facts) + (1 if revision else 0)
    body = [rng.choice(FILLER) for _ in range(max(8, int((target_tokens - fact_len * fixed) / filler_len)))]
    for text in competitors:
        body.insert(rng.randrange(len(body) + 1), text)
    at = min(len(body), max(1, int(len(body) * depth)))
    for j, fact in enumerate(facts):
        body.insert(at + j, fact)
    if revision:                                # the revision lands after the stale value
        body.insert(min(len(body), at + len(facts) + max(3, (len(body) - at) // 2)), revision)

    # Size check on the real prompt; trim filler until it fits.
    keep = set(competitors) | set(facts) | ({revision} if revision else set())
    while True:
        prompt = " ".join(body) + ask
        n = len(tok(prompt).input_ids)
        if n <= target_tokens:
            return prompt, gold, n
        spare = [i for i, s in enumerate(body) if s not in keep]
        if not spare:
            raise ValueError(f"{probe}: the facts alone take {n} tokens, over {target_tokens}")
        for i in sorted(rng.sample(spare, min(len(spare), int((n - target_tokens) / filler_len) + 1)),
                        reverse=True):
            body.pop(i)
