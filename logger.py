from rich.logging import RichHandler
import logging

log = logging.getLogger("great_library")
log.setLevel(logging.DEBUG)
if not log.handlers:
    log.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
