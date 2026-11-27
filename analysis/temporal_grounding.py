"""Temporal grounding of a compressed memory: write-time normalisation of time, the same for
every question.

The reader sees sessions dated only relatively ("[Earlier session, 26 days before this
question]") and turns full of relative times ("three weeks ago", "last Saturday"), and it does
not know today's date or weekday. Temporal and multi-session failures of CPC-Mem are mostly
date arithmetic on top of that: "244 days ago + 3 weeks prior", "last Saturday" with no
weekday, a month counted from the wrong anchor. Here every date is resolved once,
deterministically, the way a TIMEX normaliser (SUTime, HeidelTime) does:

  - the context opens with today's date and weekday (the question's date);
  - each session header, and each raw recent turn's "(from N days before ...)" prefix, gains
    the absolute date and weekday (question date minus N calendar days, exactly how N is made);
  - relative expressions inside a session get the date they point to in brackets, anchored on
    that session's date: "three weeks ago [~2023/04/29]", "last Saturday [Sat 2023/05/13]",
    "yesterday [Fri 2023/05/19]", "last month [April 2023]". Vague ones ("a few weeks ago")
    are left alone.

    python -m analysis.temporal_grounding --log <compress log> --out <compress log>_dated.jsonl
"""

import argparse
import json
import re
from datetime import date, datetime, timedelta

PREFIX = "Relevant earlier context:\n"
HEADER = re.compile(r"\[Earlier session, (\d+) days before this question\]")
RAW = re.compile(r"^\(from (\d+) days before this question\) ")
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
NUM = r"\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|(?:a )?couple(?: of)?"
EXPR = re.compile(
    rf"\b(?:(?P<n>{NUM})\s+(?P<unit>day|week|month|year)s?\s+ago"
    r"|(?P<yday>yesterday)|(?P<today>today)"
    r"|last\s+(?P<wd>monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"|last\s+(?P<last>week|weekend|month|year))\b(?!\s*\[)",
    re.IGNORECASE,
)


def day(d: date) -> str:
    return f"{d.strftime('%a')} {d.strftime('%Y/%m/%d')}"


def resolve(match: re.Match, anchor: date) -> str | None:
    if match["n"]:
        count = NUMBERS.get(match["n"].lower()) or (int(match["n"]) if match["n"].isdigit() else None)
        if not count or count > 100:  # "1000 years ago" is not a date
            return None
        unit = match["unit"].lower()
        target = anchor - timedelta(days=count * UNIT_DAYS[unit])
        return day(target) if unit == "day" else f"~{target.strftime('%Y/%m/%d')}"
    if match["yday"]:
        return day(anchor - timedelta(days=1))
    if match["today"]:
        return day(anchor)
    if match["wd"]:
        back = (anchor.weekday() - WEEKDAYS.index(match["wd"].lower())) % 7 or 7
        return day(anchor - timedelta(days=back))
    last = match["last"].lower()
    if last == "week":
        monday = anchor - timedelta(days=anchor.weekday() + 7)
        return f"week of {day(monday)}"
    if last == "weekend":
        saturday = anchor - timedelta(days=(anchor.weekday() - 5) % 7 or 7)
        return f"{saturday.strftime('%Y/%m/%d')}-{(saturday + timedelta(days=1)).strftime('%d')}"
    if last == "month":
        first = anchor.replace(day=1) - timedelta(days=1)
        return f"{MONTHS[first.month - 1]} {first.year}"
    return str(anchor.year - 1)


def ground(text: str, anchor: date) -> str:
    def sub(match):
        value = resolve(match, anchor)
        return f"{match.group(0)} [{value}]" if value else match.group(0)

    return EXPR.sub(sub, text)


def ground_context(text: str, today: date) -> str:
    """The compressed block: dated headers, and each session's text grounded on its own date."""
    body = text[len(PREFIX):] if text.startswith(PREFIX) else text
    out, anchor, last = [], None, 0
    for match in HEADER.finditer(body):
        segment = body[last : match.start()]
        out.append(ground(segment, anchor) if anchor else segment)
        anchor = today - timedelta(days=int(match[1]))
        out.append(f"[Earlier session, {day(anchor)}, {match[1]} days before this question]")
        last = match.end()
    tail = body[last:]
    out.append(ground(tail, anchor) if anchor else tail)
    return f"{PREFIX}(Today is {day(today)}.)\n" + "".join(out)


def ground_turn(text: str, today: date) -> str:
    match = RAW.match(text)
    if not match:
        return text
    anchor = today - timedelta(days=int(match[1]))
    return f"(from {day(anchor)}, {match[1]} days before this question) " + ground(text[match.end():], anchor)


def dated_history(history: list[dict], question_date: str) -> list[dict]:
    today = datetime.strptime(question_date.split(" (")[0], "%Y/%m/%d").date()
    out = []
    for index, turn in enumerate(history):
        turn = dict(turn)
        if index == 0 and turn.get("kind") == "compressed_context":
            turn["text"] = ground_context(turn["text"], today)
        elif index < len(history) - 1:
            turn["text"] = ground_turn(turn["text"], today)
        out.append(turn)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from eval.longmemeval import load_longmemeval

    dates = {item["question_id"]: item["question_date"] for item in load_longmemeval()}
    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in open(args.log, encoding="utf-8"):
            row = json.loads(line)
            if row.get("event") == "run_start":
                row["view"] = "temporal_grounding"
            elif row.get("event") == "result" and row.get("history"):
                row["history"] = dated_history(row["history"], dates[row["question_id"]])
                count += 1
            out.write(json.dumps(row) + "\n")
    print(f"grounded {count} questions -> {args.out}")


if __name__ == "__main__":
    main()
