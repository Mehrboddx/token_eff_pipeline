"""Write entity keys for every unit of a benchmark's histories (core/entities.py).

Runs once per dataset and source, before --stage compress with
--entity-keys. Units are rebuilt exactly as the pipeline builds them
(seed_memory_from_haystack -> prepare_units), so keys line up by
unit_key(unit). Already-written units are skipped, so a run can resume.

  --source ner   spaCy en_core_web_sm on CPU.
  --source llm   a small local instruct model (default Qwen2.5-1.5B-Instruct)
                 on the GPU, one short generation per sentence, given the
                 speaker, the session date and the sentence before it.

Usage (GPU machine, in the eval image):
    python -m eval.entity_keys --dataset locomo --source ner
    python -m eval.entity_keys --dataset locomo --source llm --batch 64
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import tiktoken

from core.entities import doc_keys, existing_keys, nlp, unit_key
from core.units import prepare_units, role_label
from eval.longmemeval import load_items, seed_memory_from_haystack

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENCODING = tiktoken.get_encoding("cl100k_base")
DEFAULT_LLM = "Qwen/Qwen2.5-1.5B-Instruct"

LLM_SYSTEM = """You extract facts about people from one sentence of a conversation.
Resolve "I", "me", "my" to the speaker, and other pronouns using the previous sentence.
Write one fact per line as: entity | category | value
The category is a short general word or phrase, for example: child, parent, sibling, partner, relationship status, friend, pet, occupation, workplace, education, home, hometown, origin, hobby, instrument, sport, possession, purchase, event, trip, health, preference, goal, plan, achievement, belief, inspiration.
Keep values short (a name, a thing, a place). Write NONE if the sentence states no fact about anyone."""

# Invented sentences -- never taken from a benchmark, so no test sentence can
# be answered by copying an example.
LLM_EXAMPLES = [
    ("Speaker: Priya\nPrevious: The cake looks amazing!\nSentence: Thanks, I baked it for my nephew Omar's graduation party on Saturday.",
     "Priya | nephew | Omar\nOmar | event | graduation party"),
    ("Speaker: Tom\nPrevious: How are things since the move?\nSentence: Honestly, living alone after the divorce is quiet, but I like my new job at the bakery.",
     "Tom | relationship status | divorced\nTom | home | lives alone\nTom | occupation | works at a bakery"),
    ("Speaker: Lena\nPrevious: Hey, how was your week?\nSentence: Yeah, it was fine, thanks for asking.",
     "NONE"),
]


def haystack_units(item, min_words):
    memory = seed_memory_from_haystack(item)
    entries = memory.get_sentence_entries()
    count = lambda text: len(ENCODING.encode(text, disallowed_special=()))
    return prepare_units(entries, count, min_words=min_words)


def collect(dataset, min_words, done):
    """Unique (key, speaker, text, previous text, time) over all histories."""
    seen_haystacks = set()
    todo = {}
    items = load_items(dataset)
    for number, item in enumerate(items, 1):
        # Fingerprint without the per-question evidence flags (has_answer), so
        # LoCoMo's 1,540 questions over 10 conversations seed 10 histories.
        sessions = [[{k: v for k, v in turn.items() if k != "has_answer"} for turn in session]
                    for session in item["haystack_sessions"]]
        digest = hashlib.sha1(json.dumps(sessions, sort_keys=True).encode()).hexdigest()
        if digest in seen_haystacks:
            continue
        seen_haystacks.add(digest)
        units = haystack_units(item, min_words)
        previous = {}
        for unit in units:
            key = unit_key(unit)
            if key not in done and key not in todo:
                todo[key] = {
                    "speaker": role_label(unit).rstrip(":"),
                    "text": unit.text,
                    "previous": previous.get(unit.entry_index, ""),
                    "time": unit.time or "",
                }
            previous[unit.entry_index] = unit.text
        if number % 50 == 0:
            print(f"  scanned {number}/{len(items)} questions, {len(seen_haystacks)} histories, "
                  f"{len(todo)} new units", flush=True)
    return todo


def run_ner(todo, keys, save):
    names = list(todo)
    docs = nlp().pipe((todo[key]["text"] for key in names), batch_size=256)
    for number, (key, doc) in enumerate(zip(names, docs), 1):
        keys[key] = doc_keys(doc, speaker=todo[key]["speaker"])
        if number % 20000 == 0:
            save()
            print(f"  ner {number}/{len(names)}", flush=True)


_LINE = re.compile(r"^\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*$")


def facts_to_keys(text, speaker):
    """Lemmatized keys from the model's 'entity | category | value' lines."""
    terms, chunks, entities = set(), set(), set()
    facts = [match.groups() for match in map(_LINE.match, (text or "").splitlines()) if match]
    for entity, category, value in facts:
        for part, is_entity in ((entity, True), (category, False), (value, True)):
            doc = nlp()(part)
            lemmas = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
            terms.update(lemmas)
            if lemmas:
                phrase = " ".join(lemmas)
                chunks.add(phrase)
                if is_entity:
                    entities.add(phrase)
    if facts and speaker:
        terms.add(speaker.lower())
        entities.add(speaker.lower())
    return {"t": sorted(terms), "c": sorted(chunks | entities), "e": sorted(entities)}, facts


def run_llm(todo, keys, save, model_name, batch, max_new_tokens, facts_out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    def prompt(record):
        messages = [{"role": "system", "content": LLM_SYSTEM}]
        for user, answer in LLM_EXAMPLES:
            messages += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
        user = (f"Speaker: {record['speaker']}\n"
                + (f"Previous: {record['previous']}\n" if record["previous"] else "")
                + f"Sentence: {record['text']}")
        messages.append({"role": "user", "content": user})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    names = list(todo)
    started = time.time()
    with open(facts_out, "a", encoding="utf-8") as facts_file:
        for start in range(0, len(names), batch):
            chunk = names[start:start + batch]
            encoded = tokenizer([prompt(todo[key]) for key in chunk], return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            replies = tokenizer.batch_decode(output[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
            for key, reply in zip(chunk, replies):
                keys[key], facts = facts_to_keys(reply, todo[key]["speaker"])
                facts_file.write(json.dumps({"key": key, "speaker": todo[key]["speaker"], "text": todo[key]["text"],
                                             "facts": facts}) + "\n")
            done = start + len(chunk)
            if (start // batch) % 20 == 0:
                rate = done / (time.time() - started)
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"  llm {done}/{len(names)}  {rate:.1f} units/s  peak GPU {peak:.1f} GiB", flush=True)
                save()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="locomo", choices=["longmemeval_s", "locomo"])
    parser.add_argument("--source", required=True, choices=["ner", "llm"])
    parser.add_argument("--model", default=DEFAULT_LLM, help="--source llm: Hugging Face instruct model.")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-unit-words", type=int, default=4, help="Must match the compress run (default 4).")
    parser.add_argument("--limit-units", type=int, default=None, help="Smoke test: only this many new units.")
    parser.add_argument("--out", default=None, help="Default logs/entity_keys/<dataset>_<source>.json")
    args = parser.parse_args()

    out = Path(args.out or f"logs/entity_keys/{args.dataset}_{args.source}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = existing_keys(str(out))
    print(f"{len(keys)} units already keyed in {out}")

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(keys), encoding="utf-8")
        tmp.replace(out)

    todo = collect(args.dataset, args.min_unit_words, keys)
    if args.limit_units:
        todo = dict(list(todo.items())[: args.limit_units])
    print(f"{len(todo)} units to key with {args.source}", flush=True)
    started = time.time()
    if args.source == "ner":
        run_ner(todo, keys, save)
    else:
        run_llm(todo, keys, save, args.model, args.batch, args.max_new_tokens,
                out.with_name(out.stem + "_facts.jsonl"))
    save()
    print(f"done: {len(keys)} units keyed in {out} ({time.time() - started:.0f} s)")


if __name__ == "__main__":
    main()
