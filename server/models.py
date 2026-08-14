"""Conversation state with a hard context budget (Phase 15).

The failure mode this prevents: resending the whole conversation each turn until
the prompt no longer fits the 8k window, at which point either the system prompt
or the retrieved context silently falls off the front and the assistant starts
inventing facts.

Strategy:

* keep the last ``max_recent_turns`` turns verbatim;
* compress everything older into a short extractive summary that preserves only
  what matters for support - the product under discussion, the customer's
  unresolved question, and any commitment already made;
* enforce a token budget over the whole assembled prompt, dropping *oldest
  first*, never the system prompt and never the retrieved context.

Privacy (§35): storage is in-memory by default, entries expire after
``ttl_seconds``, and nothing is written to disk unless
``persist_conversations`` is explicitly enabled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from preprocessing.khmer_detection import TextLanguage, detect_language
from preprocessing.language_mixing import SpanKind, extract_protected_spans
from rag.chunking import estimate_tokens

__all__ = ["Turn", "Conversation", "ConversationStore", "summarise_turns"]

Role = Literal["user", "assistant"]


@dataclass(slots=True)
class Turn:
    role: Role
    content: str
    created_at: float = field(default_factory=time.time)
    intent: str = ""
    grounded: bool = True
    sources: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)


def _detected_product(text: str) -> str | None:
    """First model number / SKU mentioned - the support context that matters most."""
    for span in extract_protected_spans(text):
        if span.kind in (SpanKind.MODEL_NUMBER, SpanKind.SKU):
            return span.text.strip().rstrip(".,;:)")
    return None


def summarise_turns(turns: list[Turn], *, max_chars: int = 600) -> str:
    """Extractive Khmer-safe summary of older turns.

    Deliberately *extractive*: an abstractive summary would need a second model
    call on every turn (doubling latency and cost) and would itself be a
    hallucination surface.  Pulling the concrete facts out of the transcript
    cannot introduce a new claim.
    """
    if not turns:
        return ""

    products: list[str] = []
    questions: list[str] = []
    commitments: list[str] = []

    for turn in turns:
        product = _detected_product(turn.content)
        if product and product not in products:
            products.append(product)
        if turn.role == "user":
            text = turn.content.strip()
            if "?" in text or "តើ" in text or "ប៉ុន្មាន" in text:
                questions.append(text[:120])
        elif any(
            marker in turn.content
            for marker in ("នឹងបញ្ជូន", "នឹងទាក់ទង", "បានបញ្ជូនទៅ", "escalat")
        ):
            commitments.append(turn.content.strip()[:120])

    parts: list[str] = []
    if products:
        parts.append("ផលិតផលដែលកំពុងពិភាក្សា៖ " + ", ".join(products[:3]))
    if questions:
        parts.append("សំណួរមុនៗ៖ " + " | ".join(questions[-2:]))
    if commitments:
        parts.append("ការសន្យា៖ " + commitments[-1])

    summary = "\n".join(parts)
    return summary[:max_chars]


@dataclass(slots=True)
class Conversation:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    detected_product: str | None = None
    language: str = "km"
    last_intent: str = ""
    unresolved_question: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    escalated: bool = False
    failed_attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_user(self, content: str, *, intent: str = "") -> Turn:
        turn = Turn(role="user", content=content, intent=intent)
        self.turns.append(turn)
        self.updated_at = turn.created_at
        product = _detected_product(content)
        if product:
            self.detected_product = product
        language, _ = detect_language(content, khmer_present=0.10)
        if language is TextLanguage.ENGLISH:
            self.language = "en"
        elif language in (TextLanguage.KHMER, TextLanguage.KHMER_ENGLISH):
            self.language = "km"
        if intent:
            self.last_intent = intent
        if "?" in content or "តើ" in content:
            self.unresolved_question = content.strip()[:200]
        return turn

    def add_assistant(
        self,
        content: str,
        *,
        grounded: bool = True,
        sources: list[str] | None = None,
        escalated: bool = False,
    ) -> Turn:
        turn = Turn(
            role="assistant", content=content, grounded=grounded, sources=sources or []
        )
        self.turns.append(turn)
        self.updated_at = turn.created_at
        if escalated:
            self.escalated = True
        if not grounded:
            self.failed_attempts += 1
        else:
            self.unresolved_question = ""
        return turn

    def recent(self, max_turns: int) -> list[Turn]:
        return self.turns[-max_turns:] if max_turns > 0 else []

    def compress(self, *, max_recent_turns: int = 6) -> None:
        """Fold everything older than ``max_recent_turns`` into ``summary``."""
        if len(self.turns) <= max_recent_turns:
            return
        older = self.turns[:-max_recent_turns]
        addition = summarise_turns(older)
        if addition:
            merged = f"{self.summary}\n{addition}".strip() if self.summary else addition
            self.summary = merged[-1200:]
        self.turns = self.turns[-max_recent_turns:]

    def build_history(
        self, *, max_turns: int, token_budget: int
    ) -> list[dict[str, str]]:
        """Chat-format history that fits ``token_budget``, newest kept first.

        Returns oldest-to-newest, which is what the chat template expects, but
        drops from the *oldest* end so the customer's latest question is never
        the message that falls out of the window.
        """
        selected: list[Turn] = []
        used = 0
        for turn in reversed(self.recent(max_turns)):
            cost = turn.tokens + 8  # role/formatting overhead
            if selected and used + cost > token_budget:
                break
            selected.append(turn)
            used += cost
        return [{"role": t.role, "content": t.content} for t in reversed(selected)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.turns),
            "summary": self.summary,
            "detected_product": self.detected_product,
            "language": self.language,
            "last_intent": self.last_intent,
            "escalated": self.escalated,
            "failed_attempts": self.failed_attempts,
            "age_seconds": round(time.time() - self.created_at, 1),
        }


class ConversationStore:
    """Thread-safe in-memory store with TTL and a bounded size.

    Bounded on purpose: an unbounded dict keyed by a client-supplied
    ``conversation_id`` is a trivial memory-exhaustion vector.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 3600,
        max_turns: int = 12,
        max_conversations: int = 5_000,
        enabled: bool = True,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_conversations = max_conversations
        self.enabled = enabled
        self._data: dict[str, Conversation] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, conversation_id: str) -> Conversation | None:
        if not self.enabled:
            return None
        with self._lock:
            conversation = self._data.get(conversation_id)
            if conversation is None:
                return None
            if time.time() - conversation.updated_at > self.ttl_seconds:
                del self._data[conversation_id]
                return None
            return conversation

    def get_or_create(self, conversation_id: str) -> Conversation:
        if not self.enabled:
            return Conversation(conversation_id=conversation_id)
        with self._lock:
            existing = self.get(conversation_id)
            if existing is not None:
                return existing
            self._evict_if_needed()
            conversation = Conversation(conversation_id=conversation_id)
            self._data[conversation_id] = conversation
            return conversation

    def save(self, conversation: Conversation) -> None:
        if not self.enabled:
            return
        conversation.compress(max_recent_turns=self.max_turns)
        with self._lock:
            self._data[conversation.conversation_id] = conversation

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            return self._data.pop(conversation_id, None) is not None

    def purge_expired(self) -> int:
        """Drop expired conversations.  Called by the background sweeper."""
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired = [k for k, v in self._data.items() if v.updated_at < cutoff]
            for key in expired:
                del self._data[key]
        return len(expired)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_if_needed(self) -> None:
        """Evict the least-recently-updated conversations when at capacity."""
        if len(self._data) < self.max_conversations:
            return
        self.purge_expired()
        overflow = len(self._data) - self.max_conversations + 1
        if overflow <= 0:
            return
        oldest = sorted(self._data.items(), key=lambda kv: kv[1].updated_at)[:overflow]
        for key, _ in oldest:
            del self._data[key]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "conversations": len(self._data),
                "escalated": sum(1 for c in self._data.values() if c.escalated),
                "ttl_seconds": self.ttl_seconds,
                "capacity": self.max_conversations,
            }
