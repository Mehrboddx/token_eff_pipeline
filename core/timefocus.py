"""Date-aware focus: read the time a question points to ("10 days ago",
"in the past two weeks", "on 3 June, 2023") and find the sessions inside
that window, so selection can favour them. CPC scores sentences by meaning
only; a sentence like "I just got a smoker today" never mentions the
"kitchen appliance ... 10 days ago" it answers, but its session date does.

Session times come from the unit labels the pipeline already renders:
"N days before this question" (LongMemEval) or "1:14 pm on 25 May, 2023"
(LoCoMo). Rule-based, no model calls."""

import re
from datetime import date, datetime
from typing import Optional, Tuple

_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a few": 3}
_NUM = r"(\d+|a couple of|a few|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
_MONTHS = {name: index for index, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december)"

_AGO = re.compile(rf"\b{_NUM}\s+(day|week|month|year)s?\s+ago\b", re.IGNORECASE)
_PAST = re.compile(rf"\b(?:past|last)\s+(?:{_NUM}\s+)?(day|week|month|year)s?\b", re.IGNORECASE)
_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)
_DAY_MONTH_YEAR = re.compile(rf"\b(\d{{1,2}})\s+{_MONTH},?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_DAY_YEAR = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE)
_DAYS_BEFORE = re.compile(r"(\d+) days before this question")


def _number(text: Optional[str]) -> int:
    if not text:
        return 1
    text = text.lower()
    return int(text) if text.isdigit() else _NUMBERS.get(text, 1)


def question_window(question: str) -> Optional[Tuple[str, int, int]]:
    """("relative", low, high): days before the question; ("absolute", d, d):
    an ordinal date; None when the question names no time."""
    for pattern, day_first in ((_DAY_MONTH_YEAR, True), (_MONTH_DAY_YEAR, False)):
        match = pattern.search(question)
        if match:
            if day_first:
                day, month, year = match.group(1), match.group(2), match.group(3)
            else:
                month, day, year = match.group(1), match.group(2), match.group(3)
            try:
                ordinal = date(int(year), _MONTHS[month.lower()], int(day)).toordinal()
            except ValueError:
                continue
            return ("absolute", ordinal, ordinal)
    match = _AGO.search(question)
    if match:
        days = _number(match.group(1)) * _UNIT_DAYS[match.group(2).lower()]
        slack = max(1, days // 4)  # "four weeks ago" is rarely exactly 28 days
        return ("relative", max(0, days - slack), days + slack)
    match = _PAST.search(question)
    if match:
        days = _number(match.group(1)) * _UNIT_DAYS[match.group(2).lower()]
        return ("relative", 0, days + max(1, days // 7))
    if _YESTERDAY.search(question):
        return ("relative", 0, 2)
    return None


def session_day(time_label: Optional[str]) -> Optional[Tuple[str, int]]:
    """("relative", days before the question) or ("absolute", ordinal day)."""
    if not time_label:
        return None
    match = _DAYS_BEFORE.search(time_label)
    if match:
        return ("relative", int(match.group(1)))
    match = _DAY_MONTH_YEAR.search(time_label)
    if match:
        try:
            return ("absolute", date(int(match.group(3)), _MONTHS[match.group(2).lower()],
                                     int(match.group(1))).toordinal())
        except ValueError:
            return None
    return None


def in_window(window: Tuple[str, int, int], day: Optional[Tuple[str, int]], absolute_slack: int = 1) -> bool:
    if window is None or day is None or window[0] != day[0]:
        return False
    low, high = window[1], window[2]
    if window[0] == "absolute":
        low, high = low - absolute_slack, high + absolute_slack
    return low <= day[1] <= high
