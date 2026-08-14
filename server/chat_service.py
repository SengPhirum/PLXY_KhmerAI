"""Prompt assembly and the chat orchestration pipeline.

Prompt contract (§31) - built from named sections, never string concatenation::

    [SYSTEM POLICY] [LANGUAGE POLICY] [CUSTOMER-SERVICE POLICY]
    [GROUNDING POLICY] [SECURITY POLICY]            <- prompts/system_km.md
    [CONVERSATION SUMMARY]                          <- server/models.py
    [RETRIEVED COMPANY CONTEXT]                     <- rag/citations.py (delimited)
    [CURRENT USER MESSAGE]
    [OUTPUT FORMAT]                                 <- prompts/rag_answer.md

Request pipeline::

    input guard -> conversation load -> retrieval (if required)
                -> prompt build -> generate -> output guard -> persist -> respond

The output guard runs on the *complete* answer, which is why streaming buffers
the text as it is emitted: tokens go to the client immediately for perceived
latency, and if the finished answer fails validation the stream ends with a
``done`` event carrying the corrected answer plus ``escalation_required``.
That trade-off is deliberate and documented in ``docs/architecture.md``.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.logging import get_logger
from common.paths import PROJECT_ROOT
from rag.chunking import estimate_tokens
from rag.citations import build_context_block
from rag.schemas import RetrievalConfidence, RetrievalFilters, RetrievalResult
from server.config import Settings
from server.guardrails import InputGuard, InputVerdict, OutputGuard
from server.models import Conversation, ConversationStore
from server.ollama_client import (
    CapacityExceeded,
    GenerationResult,
    OllamaClient,
    OllamaError,
)
from server.rag_service import RagService
from server.schemas import ChatRequest, ChatResponse, EscalationReason, SourceRef, StreamEvent

log = get_logger(__name__)

__all__ = ["ChatOutcome", "ChatService", "PromptBuilder"]

_SECTION_RE = re.compile(r"^#\s*\[([A-Z /-]+)\]\s*$", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Headroom for the chat template's own tokens (role markers, separators) plus a
# margin for the estimator's error against the real tokenizer.
_TEMPLATE_MARGIN = 300
_MAX_TRUNCATION_PASSES = 6
_MIN_KEPT_MESSAGE_CHARS = 200
# The Khmer system prompt is a policy document and is genuinely large (~3.6k
# estimated tokens). Above this share of the window there is not enough room
# left for retrieved context, and the assistant degrades to ungrounded answers.
_SYSTEM_PROMPT_MAX_WINDOW_SHARE = 0.55

_UNKNOWN_MARKERS = (
    "មិនមានព័ត៌មាន",
    "ខ្ញុំមិនដឹង",
    "មិនអាចបញ្ជាក់",
    "មិនមានក្នុងឯកសារ",
    "i don't have",
    "i do not have",
    "cannot confirm",
)


class PromptBuilder:
    """Assembles the runtime prompt from versioned Markdown sources."""

    def __init__(self, settings: Settings, *, prompt_dir: Path | None = None) -> None:
        self.settings = settings
        self.prompt_dir = prompt_dir or (PROJECT_ROOT / "prompts")
        self._cache: dict[str, str] = {}

    def _read(self, name: str) -> str:
        if name not in self._cache:
            path = self.prompt_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"prompt file missing: {path}")
            self._cache[name] = _HTML_COMMENT.sub("", path.read_text(encoding="utf-8")).strip()
        return self._cache[name]

    def clear_cache(self) -> None:
        self._cache.clear()

    def system_prompt(self, language: str) -> str:
        """The [SYSTEM POLICY]..[SECURITY POLICY] block, with placeholders filled."""
        source = self._read("system_en.md" if language == "en" else "system_km.md")
        return source.replace("{{company_name}}", self.settings.company_display_name)

    def validate_budget(self) -> dict[str, Any]:
        """Check at startup that the prompts fit the configured context window.

        Discovering at 03:00 that the system prompt plus the retrieved context
        does not fit - and that the runtime has been silently truncating the
        policy section - is exactly the failure this catches.
        """
        settings = self.settings
        report: dict[str, Any] = {"num_ctx": settings.num_ctx, "warnings": []}
        for language in ("km", "en"):
            tokens = estimate_tokens(self.system_prompt(language))
            report[f"system_prompt_{language}_tokens"] = tokens
            share = tokens / max(1, settings.num_ctx)
            if share > _SYSTEM_PROMPT_MAX_WINDOW_SHARE:
                report["warnings"].append(
                    f"the {language} system prompt is {tokens} tokens, {share:.0%} of the "
                    f"{settings.num_ctx}-token window; raise KHMERAI_NUM_CTX (and "
                    f"OLLAMA_CONTEXT_LENGTH) or shorten prompts/system_{language}.md"
                )
        scaffold = estimate_tokens(self._read("rag_answer.md"))
        report["answer_scaffold_tokens"] = scaffold
        headroom = (
            settings.num_ctx
            - report["system_prompt_km_tokens"]
            - scaffold
            - settings.max_output_tokens
            - _TEMPLATE_MARGIN
        )
        report["headroom_for_context_and_history_tokens"] = headroom
        if headroom < 500:
            report["warnings"].append(
                f"only {headroom} tokens remain for retrieved context and conversation "
                "history; grounded answers will be starved of evidence"
            )
        for warning in report["warnings"]:
            log.warning("chat.prompt_budget", extra={"detail": warning})
        return report

    def escalation_block(self, reason: EscalationReason, topic: str = "") -> str:
        return (
            self._read("escalation.md")
            .replace("{{escalation_reason}}", str(reason))
            .replace("{{hotline}}", self.settings.support_hotline or "-")
            .replace("{{support_email}}", self.settings.support_email or "-")
            .replace("{{business_hours}}", self.settings.business_hours)
            .replace("{{topic}}", topic or "សំណួរនេះ")
        )

    def answer_block(
        self,
        *,
        conversation_summary: str,
        retrieved_context: str,
        user_message: str,
        detected_product: str | None,
    ) -> str:
        """The RAG answer scaffold, with empty sections removed entirely."""
        template = self._read("rag_answer.md")
        filled = (
            template.replace("{{conversation_summary}}", conversation_summary or "")
            .replace("{{retrieved_context}}", retrieved_context or "")
            .replace("{{user_message}}", user_message)
            .replace("{{detected_product}}", detected_product or "មិនទាន់ដឹង")
            .replace("{{today}}", datetime.now(UTC).date().isoformat())
        )
        return self._drop_empty_sections(filled)

    @staticmethod
    def _drop_empty_sections(text: str) -> str:
        """Remove a `# [SECTION]` whose body is blank.

        An empty section rendered as a bare header teaches the model that missing
        context is normal; removing it keeps the prompt honest about what is and
        is not available.
        """
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            return text.strip()
        keep: list[str] = [text[: matches[0].start()].strip()]
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = text[match.end() : end].strip()
            if body:
                keep.append(text[match.start() : end].rstrip())
        return "\n\n".join(part for part in keep if part).strip()

    def build_messages(
        self,
        *,
        conversation: Conversation,
        user_message: str,
        retrieval: RetrievalResult | None,
        language: str,
        escalation: EscalationReason = EscalationReason.NONE,
    ) -> tuple[list[dict[str, str]], list[SourceRef]]:
        """Return the chat messages plus the citation table for the response."""
        settings = self.settings
        context_block = ""
        sources: list[SourceRef] = []
        if retrieval is not None and not retrieval.is_empty:
            context_block, citations = build_context_block(retrieval)
            sources = [
                SourceRef(
                    document_id=c.document_id,
                    title=c.title,
                    version=c.version,
                    effective_date=c.effective_date,
                    marker=c.marker,
                    source=c.source,
                )
                for c in citations
            ]

        system = self.system_prompt(language)
        if escalation is not EscalationReason.NONE:
            system = (
                f"{system}\n\n{self.escalation_block(escalation, conversation.unresolved_question)}"
            )

        # --- context budget --------------------------------------------------
        # The window is a hard limit. If the assembled prompt exceeds it, the
        # runtime truncates from the front - dropping the system prompt, which
        # is precisely how a grounded assistant turns into a hallucinating one.
        # So the budget is enforced here, shrinking the elastic parts in order
        # of least value: history first, then retrieved chunks, then the
        # customer's own message as a last resort.
        available = settings.num_ctx - settings.max_output_tokens - _TEMPLATE_MARGIN
        system_tokens = estimate_tokens(system)
        budget_for_turn = available - system_tokens

        message_text = user_message
        chunk_count = (
            len(retrieval.chunks) if retrieval is not None and not retrieval.is_empty else 0
        )

        def _render(text: str, chunks_kept: int) -> str:
            block = context_block
            if (
                retrieval is not None
                and not retrieval.is_empty
                and chunks_kept < len(retrieval.chunks)
            ):
                trimmed = retrieval.model_copy(update={"chunks": retrieval.chunks[:chunks_kept]})
                block, _ = build_context_block(trimmed) if chunks_kept else ("", [])
            return self.answer_block(
                conversation_summary=conversation.summary,
                retrieved_context=block,
                user_message=text,
                detected_product=conversation.detected_product,
            )

        user_block = _render(message_text, chunk_count)

        # 1. Drop retrieved chunks from the tail (they are ranked, so the last
        #    ones contribute least) until the turn fits.
        while chunk_count > 0 and estimate_tokens(user_block) > budget_for_turn:
            chunk_count -= 1
            user_block = _render(message_text, chunk_count)
        if retrieval is not None and chunk_count < len(retrieval.chunks):
            log.info(
                "chat.context_trimmed",
                extra={
                    "chunks_kept": chunk_count,
                    "chunks_retrieved": len(retrieval.chunks),
                    "num_ctx": settings.num_ctx,
                },
            )
            sources = sources[:chunk_count]

        # 2. Still too long: the customer's message alone exceeds the window.
        #    Truncate it rather than let the system prompt fall out.  Iterative
        #    because the token cost of Khmer is not linear in characters, so a
        #    single ratio-based cut systematically undershoots.
        if estimate_tokens(user_block) > budget_for_turn and message_text:
            for _ in range(_MAX_TRUNCATION_PASSES):
                overshoot = estimate_tokens(user_block) - budget_for_turn
                if overshoot <= 0:
                    break
                keep = max(
                    _MIN_KEPT_MESSAGE_CHARS,
                    int(
                        len(message_text)
                        * (1.0 - min(0.9, overshoot / max(1, estimate_tokens(user_block))))
                        * 0.9
                    ),
                )
                if keep >= len(message_text):
                    keep = int(len(message_text) * 0.8)
                if keep < _MIN_KEPT_MESSAGE_CHARS:
                    break
                message_text = message_text[:keep]
                user_block = _render(message_text, chunk_count)
            log.warning(
                "chat.user_message_truncated",
                extra={
                    "original_chars": len(user_message),
                    "kept_chars": len(message_text),
                    "budget_tokens": budget_for_turn,
                },
            )

        # 3. Whatever remains goes to conversation history.
        history_budget = budget_for_turn - estimate_tokens(user_block)
        history = conversation.build_history(
            max_turns=settings.conversation_max_turns, token_budget=max(0, history_budget)
        )

        messages = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user_block},
        ]
        return messages, sources


@dataclass(slots=True)
class ChatOutcome:
    """Everything one turn produced, before it becomes a response or a stream."""

    conversation_id: str
    answer: str
    language: str
    sources: list[SourceRef] = field(default_factory=list)
    confidence: float = 0.0
    escalation_required: bool = False
    escalation_reason: EscalationReason = EscalationReason.NONE
    grounded: bool = True
    conflict_detected: bool = False
    intent: str = "general_inquiry"
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    retrieval: RetrievalResult | None = None
    blocked_reason: str = ""


class ChatService:
    """Orchestrates one customer turn end to end."""

    def __init__(
        self,
        *,
        settings: Settings,
        ollama: OllamaClient,
        rag: RagService | None,
        conversations: ConversationStore,
        prompts: PromptBuilder | None = None,
        input_guard: InputGuard | None = None,
        output_guard: OutputGuard | None = None,
    ) -> None:
        self.settings = settings
        self.ollama = ollama
        self.rag = rag
        self.conversations = conversations
        self.prompts = prompts or PromptBuilder(settings)
        self.input_guard = input_guard or InputGuard(max_chars=settings.max_message_chars)
        self.output_guard = output_guard or OutputGuard()

    # -- shared pipeline ----------------------------------------------------
    def _prepare(self, request: ChatRequest) -> tuple[Conversation, InputVerdict, str]:
        conversation_id = request.conversation_id or uuid.uuid4().hex[:20]
        conversation = self.conversations.get_or_create(conversation_id)
        verdict = self.input_guard.check(request.message)
        return conversation, verdict, conversation_id

    def _language_for(self, request: ChatRequest, verdict: InputVerdict) -> str:
        if request.language in ("km", "en"):
            return request.language
        return verdict.language

    async def _retrieve(
        self, request: ChatRequest, verdict: InputVerdict, conversation: Conversation
    ) -> RetrievalResult | None:
        if not self.settings.rag_enabled or self.rag is None or not self.rag.ready:
            return None
        filters = RetrievalFilters()
        product = request.product_id or conversation.detected_product
        if product:
            filters.product_id = product
        if request.category:
            filters.category = request.category
        return await self.rag.search(verdict.message, filters=filters)

    def _decide_escalation(
        self, verdict: InputVerdict, retrieval: RetrievalResult | None, conversation: Conversation
    ) -> EscalationReason:
        if verdict.escalate is not EscalationReason.NONE:
            return verdict.escalate
        if conversation.failed_attempts >= 2:
            return EscalationReason.REPEATED_FAILURE
        if retrieval is not None and retrieval.has_conflict:
            return EscalationReason.CONFLICTING_SOURCES
        if verdict.requires_grounding and retrieval is not None and retrieval.is_empty:
            return EscalationReason.NO_INFORMATION
        return EscalationReason.NONE

    def _finalise(
        self,
        *,
        conversation: Conversation,
        verdict: InputVerdict,
        retrieval: RetrievalResult | None,
        raw_answer: str,
        sources: list[SourceRef],
        language: str,
        escalation: EscalationReason,
        model: str,
        usage: dict[str, Any],
    ) -> ChatOutcome:
        chunks = retrieval.chunks if retrieval else []
        checked = self.output_guard.check(
            raw_answer,
            chunks=chunks,
            requires_grounding=verdict.requires_grounding,
            intent=verdict.intent,
        )

        answer = checked.answer or raw_answer
        if not checked.allowed:
            sources = []
            escalation = (
                checked.escalate if checked.escalate is not EscalationReason.NONE else escalation
            )

        confidence = retrieval.confidence_score if retrieval else 0.0
        if retrieval is not None and retrieval.confidence is RetrievalConfidence.HIGH:
            confidence = max(confidence, 0.8)

        conversation.add_assistant(
            answer,
            grounded=checked.grounded,
            sources=[s.document_id for s in sources],
            escalated=escalation is not EscalationReason.NONE,
        )
        self.conversations.save(conversation)

        return ChatOutcome(
            conversation_id=conversation.conversation_id,
            answer=answer,
            language=language,
            sources=sources,
            confidence=round(confidence, 4),
            escalation_required=escalation is not EscalationReason.NONE,
            escalation_reason=escalation,
            grounded=checked.grounded,
            conflict_detected=bool(retrieval and retrieval.has_conflict),
            intent=verdict.intent,
            model=model,
            usage=usage,
            retrieval=retrieval,
            blocked_reason=checked.reason,
        )

    def _refusal(
        self, conversation: Conversation, verdict: InputVerdict, language: str
    ) -> ChatOutcome:
        """Response for an input the guard refused."""
        messages = {
            "empty_message": "សូមសរសេរសំណួររបស់លោកអ្នក ដើម្បីឱ្យខ្ញុំអាចជួយបាន។",
            "message_too_long": (
                f"សារវែងពេក។ សូមសរសេរឱ្យខ្លីជាងនេះ (មិនលើសពី {self.settings.max_message_chars} តួអក្សរ)។"
            ),
            "prompt_injection": (
                "ខ្ញុំមិនអាចធ្វើតាមសំណើនោះបានទេ ប៉ុន្តែខ្ញុំរីករាយជួយឆ្លើយសំណួរអំពីផលិតផល សេវាកម្ម ការធានា និងតម្លៃរបស់យើង។"
            ),
        }
        answer = messages.get(verdict.reason, messages["empty_message"])
        return ChatOutcome(
            conversation_id=conversation.conversation_id,
            answer=answer,
            language=language,
            intent=verdict.intent,
            blocked_reason=verdict.reason,
            grounded=True,
        )

    # -- non-streaming ------------------------------------------------------
    async def complete(self, request: ChatRequest) -> ChatOutcome:
        conversation, verdict, _ = self._prepare(request)
        language = self._language_for(request, verdict)

        if not verdict.allowed:
            log.warning("chat.input_blocked", extra=verdict.to_log())
            return self._refusal(conversation, verdict, language)

        conversation.add_user(verdict.message, intent=verdict.intent)
        retrieval = await self._retrieve(request, verdict, conversation)
        escalation = self._decide_escalation(verdict, retrieval, conversation)

        messages, sources = self.prompts.build_messages(
            conversation=conversation,
            user_message=verdict.message,
            retrieval=retrieval,
            language=language,
            escalation=escalation,
        )

        options = self.settings.generation_options(
            temperature=request.temperature, num_predict=request.max_output_tokens
        )
        result: GenerationResult = await self.ollama.chat(
            messages, model=request.model, options=options
        )

        return self._finalise(
            conversation=conversation,
            verdict=verdict,
            retrieval=retrieval,
            raw_answer=result.text,
            sources=sources,
            language=language,
            escalation=escalation,
            model=result.model,
            usage=result.usage(),
        )

    # -- streaming ----------------------------------------------------------
    async def stream(
        self, request: ChatRequest, *, request_id: str = ""
    ) -> AsyncIterator[StreamEvent]:
        conversation, verdict, conversation_id = self._prepare(request)
        language = self._language_for(request, verdict)

        yield StreamEvent(type="start", conversation_id=conversation_id, request_id=request_id)

        if not verdict.allowed:
            log.warning("chat.input_blocked", extra=verdict.to_log())
            outcome = self._refusal(conversation, verdict, language)
            yield StreamEvent(type="token", conversation_id=conversation_id, content=outcome.answer)
            yield StreamEvent(
                type="done",
                conversation_id=conversation_id,
                request_id=request_id,
                content=outcome.answer,
            )
            return

        conversation.add_user(verdict.message, intent=verdict.intent)
        retrieval = await self._retrieve(request, verdict, conversation)
        escalation = self._decide_escalation(verdict, retrieval, conversation)

        messages, sources = self.prompts.build_messages(
            conversation=conversation,
            user_message=verdict.message,
            retrieval=retrieval,
            language=language,
            escalation=escalation,
        )
        options = self.settings.generation_options(
            temperature=request.temperature, num_predict=request.max_output_tokens
        )

        buffer: list[str] = []
        metrics: dict[str, Any] = {}
        started = time.perf_counter()
        try:
            async for chunk in self.ollama.chat_stream(
                messages, model=request.model, options=options
            ):
                if chunk.text:
                    buffer.append(chunk.text)
                    yield StreamEvent(
                        type="token", conversation_id=conversation_id, content=chunk.text
                    )
                if chunk.done:
                    metrics = chunk.metrics
        except CapacityExceeded as exc:
            yield StreamEvent(
                type="error",
                conversation_id=conversation_id,
                request_id=request_id,
                error=str(exc),
            )
            return
        except OllamaError as exc:
            log.error("chat.stream.ollama_error", extra={"error": str(exc)})
            yield StreamEvent(
                type="error",
                conversation_id=conversation_id,
                request_id=request_id,
                error="generation_failed",
            )
            return

        raw_answer = "".join(buffer)
        outcome = self._finalise(
            conversation=conversation,
            verdict=verdict,
            retrieval=retrieval,
            raw_answer=raw_answer,
            sources=sources,
            language=language,
            escalation=escalation,
            model=str(metrics.get("model", self.settings.ollama_model)),
            usage={k: v for k, v in metrics.items() if k not in ("model", "done_reason")}
            or {"total_duration_ms": round((time.perf_counter() - started) * 1000, 1)},
        )

        if outcome.sources:
            yield StreamEvent(
                type="sources", conversation_id=conversation_id, sources=outcome.sources
            )

        yield StreamEvent(
            type="done",
            conversation_id=conversation_id,
            request_id=request_id,
            # If the guard replaced the answer, the client must render the
            # corrected text rather than the tokens it already displayed.
            content=outcome.answer if outcome.answer != raw_answer else "",
            sources=outcome.sources,
            confidence=outcome.confidence,
            escalation_required=outcome.escalation_required,
            escalation_reason=outcome.escalation_reason,
            usage=outcome.usage,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def is_unknown_answer(answer: str) -> bool:
        lowered = answer.lower()
        return any(marker in answer or marker in lowered for marker in _UNKNOWN_MARKERS)

    def to_response(
        self, outcome: ChatOutcome, *, request_id: str, versions: dict[str, str]
    ) -> ChatResponse:
        return ChatResponse(
            conversation_id=outcome.conversation_id,
            answer=outcome.answer,
            language=outcome.language,
            sources=outcome.sources,
            confidence=outcome.confidence,
            escalation_required=outcome.escalation_required,
            escalation_reason=outcome.escalation_reason,
            grounded=outcome.grounded,
            conflict_detected=outcome.conflict_detected,
            intent=outcome.intent,
            model=outcome.model or self.settings.ollama_model,
            prompt_version=versions.get("prompt", ""),
            index_version=(
                outcome.retrieval.index_version
                if outcome.retrieval
                else versions.get("knowledge_index", "")
            ),
            request_id=request_id,
            usage=outcome.usage,
        )
