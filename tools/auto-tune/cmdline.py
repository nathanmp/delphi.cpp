"""Command-line editing for llama-server argument lists."""
from __future__ import annotations

from typing import Sequence


def option_value(command: Sequence[str], option: str) -> str | None:
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if option.startswith("--") and item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None


def remove_option(command: Sequence[str], option: str, *, has_value: bool = True) -> tuple[str, ...]:
    out: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item == option:
            skip = has_value
            continue
        if has_value and option.startswith("--") and item.startswith(option + "="):
            continue
        out.append(item)
    return tuple(out)


def set_option(command: Sequence[str], option: str, value: str | None) -> tuple[str, ...]:
    """Replace the value of *option*, append the option if it is missing, or remove it when *value* is None."""
    if value is None:
        return remove_option(command, option)
    out = list(command)
    for index, item in enumerate(out):
        if item == option and index + 1 < len(out):
            out[index + 1] = value
            return tuple(out)
        if option.startswith("--") and item.startswith(option + "="):
            out[index] = f"{option}={value}"
            return tuple(out)
    return (*out, option, value)


def first_value(command: Sequence[str], *options: str) -> str | None:
    """The value of the first of *options* (aliases of one option) that the command has."""
    for option in options:
        value = option_value(command, option)
        if value is not None:
            return value
    return None


# speculative strategy options, which the tuner sets; the draft model and its placement (--spec-draft-model, -md,
# --spec-draft-device, --spec-draft-ngl, --spec-draft-threads, ...) are not among them
SPEC_STRATEGY = ("--spec-type", "--spec-default", "--spec-draft-adaptive", "--no-spec-draft-adaptive",
                 "--spec-draft-adaptive-mode", "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-n-min-adaptive",
                 "--spec-draft-n-start", "--spec-draft-p-min", "--spec-draft-p-split", "--draft-max", "--draft-min",
                 "--draft-n", "--draft-n-min", "--draft-p-min", "--draft-p-split")
SPEC_SWITCHES = ("--spec-default", "--spec-draft-adaptive", "--no-spec-draft-adaptive")


def is_spec_strategy(name: str) -> bool:
    return name in SPEC_STRATEGY or name.startswith(("--spec-ngram-", "--spec-synth-"))


def strip_spec(command: Sequence[str]) -> tuple[str, ...]:
    """The command without its speculative strategy options."""
    out: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        name = item.split("=", 1)[0]
        if is_spec_strategy(name):
            skip = "=" not in item and name not in SPEC_SWITCHES
            continue
        out.append(item)
    return tuple(out)


def spec_args(spec: dict[str, str | None]) -> list[str]:
    """--spec-* arguments of a strategy: option -> value, None for a switch."""
    out: list[str] = []
    for option, value in spec.items():
        out.append(option)
        if value is not None:
            out.append(value)
    return out
