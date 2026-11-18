import io
import sys


def utf8_console() -> None:
    """Print UTF-8 on consoles that default to another encoding (Windows),
    replacing what the console cannot show instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper) and stream.encoding.lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")
