"""Plain data types for the store data that the agent uses."""

from dataclasses import dataclass, field


@dataclass
class Variant:
    id: str
    title: str
    sku: str | None
    price_bdt: int
    stock: int
    specs: dict = field(default_factory=dict)


@dataclass
class Product:
    id: str
    handle: str
    title: str
    description: str
    category: str | None
    tags: list[str]
    specs: dict
    variants: list[Variant]

    @property
    def total_stock(self) -> int:
        return sum(v.stock for v in self.variants)

    @property
    def lowest_price_bdt(self) -> int:
        return min(v.price_bdt for v in self.variants)


@dataclass
class Address:
    first_name: str
    last_name: str
    address_1: str
    city: str
    postal_code: str
    phone: str
    country_code: str = "bd"


@dataclass
class OrderItem:
    title: str
    variant_title: str | None
    quantity: int


@dataclass
class TrackingStep:
    status: str
    label: str
    at: str


@dataclass
class TrackingView:
    tracking_number: str
    status: str
    label: str
    order_display_id: int | None
    history: list[TrackingStep]


@dataclass
class OrderView:
    display_id: int
    status: str
    created_at: str
    total_bdt: int
    items: list[OrderItem]
    shipping_option: str | None
    tracking: TrackingView | None


@dataclass
class Customer:
    id: str
    email: str
    first_name: str
    last_name: str
    phone: str | None
    default_address: Address | None
