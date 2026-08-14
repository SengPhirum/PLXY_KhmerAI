"""Access control for documents, metadata and admin operations.

Three distinct decisions, deliberately kept separate:

``can_retrieve``   may this caller's query see this document at all?
``can_quote``      may the answer reproduce the document's text to the customer?
``can_see_field``  may this metadata field appear in the API response?

The middle one matters more than it looks: an ``internal`` document may
legitimately *inform* an answer (a support agent's view) while never being
quotable to a customer.  Collapsing the two into one boolean is how confidential
text ends up in a customer transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from company_data.schema import AccessLevel, Confidentiality, DocumentStatus

__all__ = [
    "Principal",
    "AccessDecision",
    "can_retrieve",
    "can_quote",
    "can_see_field",
    "filter_metadata",
    "CUSTOMER",
    "AGENT",
    "ADMIN",
]


class _Rank(IntEnum):
    ANONYMOUS = 0
    CUSTOMER = 1
    AGENT = 2
    ADMIN = 3


_ACCESS_RANK: dict[str, _Rank] = {
    AccessLevel.ANONYMOUS: _Rank.ANONYMOUS,
    AccessLevel.CUSTOMER: _Rank.CUSTOMER,
    AccessLevel.AGENT: _Rank.AGENT,
    AccessLevel.ADMIN: _Rank.ADMIN,
}
# Minimum principal rank required to *see* a document at each confidentiality.
_CONFIDENTIALITY_MIN_RANK: dict[str, _Rank] = {
    Confidentiality.PUBLIC: _Rank.ANONYMOUS,
    Confidentiality.CUSTOMER_SHAREABLE: _Rank.ANONYMOUS,
    Confidentiality.INTERNAL: _Rank.AGENT,
    Confidentiality.RESTRICTED: _Rank.ADMIN,
}
# Confidentiality levels whose text may be quoted verbatim to a customer.
_QUOTABLE = frozenset({str(Confidentiality.PUBLIC), str(Confidentiality.CUSTOMER_SHAREABLE)})

# Metadata fields that must never leave the server, at any access level.
_NEVER_EXPOSED = frozenset({"source_path", "content_hash", "owner", "access_level"})
# Fields an anonymous/customer principal may see.
_CUSTOMER_VISIBLE = frozenset(
    {
        "document_id", "document_title", "product_id", "product_name",
        "service_id", "service_name", "category", "subcategory", "version",
        "effective_date", "status", "language",
    }
)


@dataclass(slots=True, frozen=True)
class Principal:
    """Who is asking."""

    identity: str = "anonymous"
    access_level: str = AccessLevel.ANONYMOUS
    scopes: frozenset[str] = field(default_factory=frozenset)

    @property
    def rank(self) -> _Rank:
        return _ACCESS_RANK.get(self.access_level, _Rank.ANONYMOUS)

    @property
    def is_customer_facing(self) -> bool:
        """True for principals whose output a customer will read."""
        return self.rank <= _Rank.CUSTOMER

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or self.rank is _Rank.ADMIN


CUSTOMER = Principal(identity="customer", access_level=AccessLevel.CUSTOMER)
AGENT = Principal(identity="agent", access_level=AccessLevel.AGENT)
ADMIN = Principal(
    identity="admin", access_level=AccessLevel.ADMIN, scopes=frozenset({"reindex", "diagnostics"})
)


@dataclass(slots=True, frozen=True)
class AccessDecision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def can_retrieve(
    principal: Principal, metadata: dict[str, Any], *, allow_non_active: bool = False
) -> AccessDecision:
    """May this principal's query match this document?"""
    confidentiality = str(metadata.get("confidentiality", Confidentiality.INTERNAL))
    required = _CONFIDENTIALITY_MIN_RANK.get(confidentiality, _Rank.ADMIN)
    if principal.rank < required:
        return AccessDecision(False, f"confidentiality={confidentiality} requires {required.name}")

    status = str(metadata.get("status", DocumentStatus.ACTIVE))
    if status != DocumentStatus.ACTIVE and not allow_non_active:
        return AccessDecision(False, f"status={status}")

    validation = str(metadata.get("validation_status", "valid"))
    if validation == "quarantined":
        return AccessDecision(False, "document is quarantined")
    return AccessDecision(True)


def can_quote(principal: Principal, metadata: dict[str, Any]) -> AccessDecision:
    """May this document's text be reproduced in an answer the customer reads?"""
    confidentiality = str(metadata.get("confidentiality", Confidentiality.INTERNAL))
    if principal.is_customer_facing and confidentiality not in _QUOTABLE:
        return AccessDecision(
            False, f"confidentiality={confidentiality} is not customer-quotable"
        )
    return can_retrieve(principal, metadata)


def can_see_field(principal: Principal, field_name: str) -> bool:
    if field_name in _NEVER_EXPOSED:
        return principal.rank >= _Rank.ADMIN
    if principal.rank >= _Rank.AGENT:
        return True
    return field_name in _CUSTOMER_VISIBLE


def filter_metadata(principal: Principal, metadata: dict[str, Any]) -> dict[str, Any]:
    """Project metadata down to what this principal is allowed to see."""
    return {k: v for k, v in metadata.items() if can_see_field(principal, k)}
