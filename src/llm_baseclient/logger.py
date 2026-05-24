import logging
from logging import Logger

from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_tracebacks


def get_logger() -> Logger:
    """Set up a logger with RichHandler for enhanced logging output."""
    install_rich_tracebacks(show_locals=True, suppress=[__file__])

    console = Console(stderr=True, highlight=True, log_time_format="[%H.%M]")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    rich_handler = RichHandler(
        console=console,
        show_time=True,  # Display timestamp
        show_level=True,  # Display log level
        show_path=True,  # Display file and line number
        enable_link_path=True,  # Make file paths clickable in supported terminals
        markup=True,  # Allow rich markup in log messages (e.g., [bold red]Error![/bold red])
        rich_tracebacks=True,  # Use rich's beautiful tracebacks for exceptions
        tracebacks_show_locals=True,  # Show local variables in tracebacks
        tracebacks_word_wrap=True,  # Wrap long lines in tracebacks
    )

    logger.addHandler(rich_handler)
    return logger


if __name__ == "__main__":
    logger = get_logger()

    # --- Demo ---
    logger.debug("This is a [cyan]debug[/cyan] message. Very detailed info.")
    logger.info("Loaded [bold green]1234[/bold green] records.")
    logger.warning("Configuration file '[yellow]settings.ini[/yellow]' not found.")
    logger.info("Visit https://example.com for more information.")

    def divide_by_zero():  # noqa
        a = 10
        b = 0
        logger.debug(f"Attempting division: {a} / {b}")
        return a / b

    try:
        result = divide_by_zero()
        logger.info(f"Result: {result}")
    except ZeroDivisionError:
        # logger.exception() automatically captures the current exception info
        logger.exception("An [bold red]critical error[/bold red] occurred during calculation!")

    logger.critical("System is shutting down due to unrecoverable error. [blink red]Emergency![/blink red]")
