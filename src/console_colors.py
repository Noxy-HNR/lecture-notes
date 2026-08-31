"""ANSI color helpers for the live console view. Using colorama's init(autoreset=True)
so plain terminals (old cmd.exe) still render color correctly on Windows, and every
colored string automatically resets style afterward instead of bleeding into later output.
"""
from colorama import init, Fore, Style

init(autoreset=True)


def timestamp(text: str) -> str:
    return f"{Fore.CYAN}{text}{Style.RESET_ALL}"


def transcript(text: str) -> str:
    return f"{Fore.WHITE}{text}{Style.RESET_ALL}"


def flag(text: str) -> str:
    """For inline transcription-error flags within transcript text."""
    return f"{Fore.YELLOW}{Style.BRIGHT}{text}{Style.RESET_ALL}"


def success(text: str) -> str:
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"


def autosave(text: str) -> str:
    return f"{Fore.MAGENTA}{text}{Style.RESET_ALL}"


def warning(text: str) -> str:
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"


def error(text: str) -> str:
    return f"{Fore.RED}{Style.BRIGHT}{text}{Style.RESET_ALL}"


def info(text: str) -> str:
    return f"{Fore.BLUE}{text}{Style.RESET_ALL}"


def dim(text: str) -> str:
    return f"{Style.DIM}{text}{Style.RESET_ALL}"


def heading(text: str) -> str:
    return f"{Fore.MAGENTA}{Style.BRIGHT}{text}{Style.RESET_ALL}"
