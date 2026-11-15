"""Conversation state: who the customer is, the messages, and pending order summaries."""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from support_agent.store.models import Address, Customer, Quote

Channel = Literal["chat", "voice"]


@dataclass
class Identity:
    kind: Literal["customer", "guest"]
    token: str | None = None
    customer: Customer | None = None

    @classmethod
    def guest(cls) -> "Identity":
        return cls(kind="guest")


@dataclass
class OrderSummary:
    id: str
    items: list[tuple[str, int]]
    address: Address
    email: str
    quote: Quote
    customer_turn_at_creation: int


@dataclass
class Session:
    id: str
    identity: Identity
    channel: Channel
    messages: list[dict] = field(default_factory=list)
    customer_turns: int = 0
    summaries: dict[str, OrderSummary] = field(default_factory=dict)
    verified_orders: set[int] = field(default_factory=set)

    @classmethod
    def new(cls, identity: Identity, channel: Channel = "chat") -> "Session":
        return cls(id=uuid.uuid4().hex, identity=identity, channel=channel)

    def start_over(self, identity: Identity) -> None:
        """Changes the customer and forgets the conversation, the summaries, and the
        verified orders, so that nothing passes from one person to the next."""
        self.identity = identity
        self.messages = []
        self.customer_turns = 0
        self.summaries = {}
        self.verified_orders = set()
