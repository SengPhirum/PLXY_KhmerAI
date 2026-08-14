"""Chat templating and loss masking for Khmer support SFT.

Two things here decide whether fine-tuning works at all:

**Template fidelity.**  Training must use *exactly* the chat template the model
will be served with, or the model learns a format it never sees at inference.
``render_conversation`` prefers the tokenizer's own
``apply_chat_template`` and only falls back to the explicit ChatML renderer when
the tokenizer has no template, which is checked and logged rather than assumed.

**Completion-only loss.**  Computing loss over the prompt teaches the model to
*generate customer questions*, which is both wasted capacity and a source of
odd behaviour.  ``build_completion_mask`` marks every prompt token as ``-100``
so loss is taken only on assistant turns - and it handles multi-turn samples,
where there are several assistant spans, not just a trailing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from common.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "ChatMLTemplate",
    "render_conversation",
    "build_completion_mask",
    "DEFAULT_SYSTEM_PROMPT_KM",
    "IGNORE_INDEX",
]

IGNORE_INDEX = -100

DEFAULT_SYSTEM_PROMPT_KM = (
    "អ្នកគឺជាជំនួយការបម្រើអតិថិជនរបស់ក្រុមហ៊ុន។ "
    "សូមឆ្លើយជាភាសាខ្មែរ ខ្លី ច្បាស់ និងគួរសម។ "
    "ប្រើតែព័ត៌មានពីឯកសារក្រុមហ៊ុនដែលបានផ្តល់ជូន។ "
    "ប្រសិនបើអ្នកមិនដឹង សូមប្រាប់ដោយស្មោះត្រង់ ហើយណែនាំឱ្យទាក់ទងបុគ្គលិក។"
)


@dataclass(slots=True, frozen=True)
class ChatMLTemplate:
    """Explicit ChatML renderer - the fallback when a tokenizer has no template."""

    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"

    def render_message(self, role: str, content: str) -> str:
        return f"{self.im_start}{role}\n{content}{self.im_end}\n"

    def render(self, messages: list[dict[str, str]], *, add_generation_prompt: bool = False) -> str:
        out = "".join(self.render_message(m["role"], m["content"]) for m in messages)
        if add_generation_prompt:
            out += f"{self.im_start}assistant\n"
        return out

    def assistant_prefix(self) -> str:
        return f"{self.im_start}assistant\n"

    def assistant_suffix(self) -> str:
        return f"{self.im_end}\n"


_DEFAULT_TEMPLATE = ChatMLTemplate()


def _ensure_system(messages: list[dict[str, str]], system_prompt: str | None) -> list[dict[str, str]]:
    if not system_prompt:
        return list(messages)
    if messages and messages[0].get("role") == "system":
        return list(messages)
    return [{"role": "system", "content": system_prompt}, *messages]


def render_conversation(
    messages: list[dict[str, str]],
    tokenizer: Any = None,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT_KM,
    add_generation_prompt: bool = False,
) -> str:
    """Render a conversation to the string the model is trained/served on."""
    full = _ensure_system(messages, system_prompt)
    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            full, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    if tokenizer is not None:
        log.warning(
            "training.chat_template.missing",
            extra={
                "tokenizer": type(tokenizer).__name__,
                "action": "falling back to explicit ChatML - verify this matches the Modelfile",
            },
        )
    return _DEFAULT_TEMPLATE.render(full, add_generation_prompt=add_generation_prompt)


def build_completion_mask(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT_KM,
    max_length: int | None = None,
) -> dict[str, list[int]]:
    """Tokenise a conversation and mask everything except the assistant turns.

    Built incrementally - render the conversation prefix up to each message, then
    up to and including it, and take the token-length difference.  That is robust
    to any chat template, including ones that insert tokens between turns, which
    a naive "find the assistant marker in the token stream" approach is not.

    Returns ``{"input_ids", "attention_mask", "labels"}`` with prompt positions
    set to ``IGNORE_INDEX``.
    """
    full = _ensure_system(messages, system_prompt)

    def _encode(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    input_ids: list[int] = []
    labels: list[int] = []
    previous_text = ""

    for index, message in enumerate(full):
        prefix = render_conversation(
            full[:index], tokenizer, system_prompt=None, add_generation_prompt=False
        )
        with_message = render_conversation(
            full[: index + 1], tokenizer, system_prompt=None, add_generation_prompt=False
        )
        # Guard against a template that is not prefix-stable.
        if not with_message.startswith(prefix):
            log.warning(
                "training.chat_template.not_prefix_stable",
                extra={"message_index": index, "role": message.get("role", "?")},
            )
        segment_text = with_message[len(prefix) :] if with_message.startswith(prefix) else with_message
        segment = _encode(segment_text)
        input_ids.extend(segment)
        if message.get("role") == "assistant":
            labels.extend(segment)
        else:
            labels.extend([IGNORE_INDEX] * len(segment))
        previous_text = with_message

    if not previous_text:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    if max_length is not None and len(input_ids) > max_length:
        # Truncate from the FRONT: the final assistant turn is the training
        # signal, so dropping the oldest context is the only safe truncation.
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def supervised_token_ratio(masked: dict[str, list[int]]) -> float:
    """Share of tokens that contribute to the loss - a dataset sanity check.

    A ratio near zero means the mask is wrong (nothing is being learned); a ratio
    near one means masking is not being applied at all.  Healthy Khmer support
    data sits around 0.25-0.5.
    """
    labels = masked.get("labels", [])
    if not labels:
        return 0.0
    return sum(1 for token in labels if token != IGNORE_INDEX) / len(labels)
