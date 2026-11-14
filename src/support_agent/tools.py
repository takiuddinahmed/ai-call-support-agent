"""The store tools that the model can call, and the rules that guard them."""

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass

from support_agent.session import OrderSummary, Session
from support_agent.specs import SpecFilter, SpecFilterError, filter_products
from support_agent.store.client import StoreError
from support_agent.store.models import Address, OrderView, Product

MAX_SEARCH_RESULTS = 8
MAX_QUANTITY = 20


class ToolError(Exception):
    """A rule failure. The message goes back to the model."""


@dataclass
class ToolResult:
    ok: bool
    content: str
    ms: float


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


ADDRESS_SCHEMA = {
    "type": "object",
    "properties": {
        "first_name": {"type": "string"},
        "last_name": {"type": "string"},
        "address_1": {"type": "string", "description": "House, road, and area."},
        "city": {"type": "string"},
        "postal_code": {"type": "string"},
        "phone": {"type": "string", "description": "Bangladesh mobile number, 01XXXXXXXXX."},
    },
}

TOOL_SPECS: list[dict] = [
    _function(
        "search_products",
        "Find products by words, category, and spec values. Use spec filters for needs such as "
        "voltage, interface, Wi-Fi, memory, or range. Returns at most 8 products.",
        {
            "text": {"type": "string", "description": "Words that must all appear, in English."},
            "category": {"type": "string"},
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Spec key, e.g. interface."},
                        "op": {"type": "string", "enum": ["eq", "gte", "lte", "contains"]},
                        "value": {"type": ["string", "number", "boolean"]},
                    },
                    "required": ["key", "op", "value"],
                },
            },
            "in_stock_only": {"type": "boolean"},
        },
        [],
    ),
    _function(
        "get_product",
        "Get one product with all variants, prices in BDT, stock, and specs.",
        {"handle": {"type": "string"}},
        ["handle"],
    ),
    _function(
        "list_my_orders",
        "List the orders of the logged-in customer with status and tracking.",
        {},
        [],
    ),
    _function(
        "get_order_status",
        "Get an order and its tracking by order number. A guest must give the email or the "
        "phone of the order.",
        {
            "display_id": {"type": "integer", "description": "The order number, e.g. 6."},
            "email": {"type": "string"},
            "phone": {"type": "string"},
        },
        ["display_id"],
    ),
    _function(
        "track_shipment",
        "Get the delivery status and history for a tracking number such as DEMO-1234567890.",
        {"tracking_number": {"type": "string"}},
        ["tracking_number"],
    ),
    _function(
        "prepare_order",
        "Check an order and get a summary with the total and a summary_id. Read the summary to "
        "the customer and ask for a clear yes before place_order. For a logged-in customer, the "
        "account address and email are used unless you give others.",
        {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "handle": {"type": "string"},
                        "variant_title": {"type": "string", "description": "e.g. 4 GB"},
                        "quantity": {"type": "integer"},
                    },
                    "required": ["handle", "quantity"],
                },
            },
            "address": ADDRESS_SCHEMA,
            "email": {"type": "string"},
        },
        ["items"],
    ),
    _function(
        "place_order",
        "Place the order of a summary from prepare_order with cash on delivery. Call it only "
        "after the customer said a clear yes to that summary.",
        {"summary_id": {"type": "string"}},
        ["summary_id"],
    ),
]


def _is_bd_mobile(phone: str) -> bool:
    digits = re.sub(r"[\s-]", "", phone).lstrip("+")
    local = digits[3:] if digits.startswith("880") else digits.removeprefix("0")
    return re.fullmatch(r"1[3-9]\d{8}", local) is not None


def _product_view(product: Product, full: bool = False) -> dict:
    view = {
        "handle": product.handle,
        "title": product.title,
        "category": product.category,
        "price_bdt": product.lowest_price_bdt,
        "stock": product.total_stock,
        "specs": product.specs,
    }
    if full or len(product.variants) > 1:
        view["variants"] = [
            {"title": v.title, "price_bdt": v.price_bdt, "stock": v.stock, "specs": v.specs}
            for v in product.variants
        ]
    if full:
        view["description"] = product.description
    return view


def _order_view(order: OrderView) -> dict:
    view = asdict(order)
    if order.tracking:
        view["tracking"] = {
            "tracking_number": order.tracking.tracking_number,
            "status": order.tracking.label,
            "history": [{"status": h.label, "at": h.at} for h in order.tracking.history],
        }
    return view


class ToolRunner:
    def __init__(self, store, session: Session):
        self.store = store
        self.session = session

    async def run(self, name: str, arguments: str) -> ToolResult:
        start = time.perf_counter()
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None or name not in {t["function"]["name"] for t in TOOL_SPECS}:
                raise ToolError(f"Unknown tool '{name}'.")
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError as error:
                raise ToolError(f"The arguments are not valid JSON: {error}.") from error
            if not isinstance(args, dict):
                raise ToolError("The arguments must be a JSON object.")
            data = await handler(**args)
            content, ok = json.dumps(data, ensure_ascii=False, separators=(",", ":")), True
        except (ToolError, SpecFilterError, StoreError) as error:
            content, ok = str(error), False
        except TypeError as error:
            content, ok = f"Wrong arguments for '{name}': {error}.", False
        return ToolResult(ok, content, (time.perf_counter() - start) * 1000)

    # Products

    async def _tool_search_products(self, text=None, category=None, filters=None,
                                    in_stock_only=False) -> dict:
        spec_filters = [SpecFilter(f.get("key"), f.get("op"), f.get("value")) for f in filters or []]
        products = filter_products(
            await self.store.list_products(), text, category, spec_filters, bool(in_stock_only)
        )
        return {
            "count": len(products),
            "products": [_product_view(p) for p in products[:MAX_SEARCH_RESULTS]],
        }

    async def _tool_get_product(self, handle: str) -> dict:
        product = await self.store.get_product(handle)
        if product is None:
            raise ToolError(f"No product has the handle '{handle}'. Use search_products.")
        return _product_view(product, full=True)

    # Orders

    async def _tool_list_my_orders(self) -> dict:
        identity = self.session.identity
        if identity.kind != "customer":
            raise ToolError(
                "The customer is not logged in. Ask for the order number and the email or phone."
            )
        orders = await self.store.list_orders(identity.token)
        self.session.verified_orders.update(o.display_id for o in orders)
        return {"orders": [_order_view(o) for o in orders]}

    async def _tool_get_order_status(self, display_id: int, email=None, phone=None) -> dict:
        identity = self.session.identity
        if identity.kind == "customer" and not email and not phone:
            email = identity.customer.email
        if not email and not phone:
            raise ToolError("Ask the customer for the email or phone of the order first.")
        order = await self.store.lookup_order(int(display_id), email, phone)
        if order is None:
            raise ToolError("No order matches this order number with this email or phone.")
        self.session.verified_orders.add(order.display_id)
        return {"order": _order_view(order)}

    async def _tool_track_shipment(self, tracking_number: str) -> dict:
        tracking = await self.store.track(tracking_number.strip().upper())
        if tracking is None:
            raise ToolError("No shipment has this tracking number.")
        return {
            "tracking_number": tracking.tracking_number,
            "status": tracking.label,
            "history": [{"status": h.label, "at": h.at} for h in tracking.history],
        }

    async def _tool_prepare_order(self, items, address=None, email=None) -> dict:
        if not items:
            raise ToolError("Give at least one item.")
        resolved = []
        for item in items:
            quantity = item.get("quantity")
            if not isinstance(quantity, int) or not 1 <= quantity <= MAX_QUANTITY:
                raise ToolError(f"The quantity must be 1 to {MAX_QUANTITY}.")
            product = await self.store.get_product(item.get("handle", ""))
            if product is None:
                raise ToolError(f"No product has the handle '{item.get('handle')}'.")
            wanted = (item.get("variant_title") or "").strip().lower()
            if len(product.variants) == 1 and not wanted:
                variant = product.variants[0]
            else:
                variant = next((v for v in product.variants if v.title.lower() == wanted), None)
                if variant is None:
                    titles = ", ".join(v.title for v in product.variants)
                    raise ToolError(f"Ask which variant of {product.title}: {titles}.")
            resolved.append((variant.id, quantity))

        final_address, final_email = self._order_contact(address or {}, email)
        quote = await self.store.quote(resolved, final_address.city)
        summary = OrderSummary(
            id=uuid.uuid4().hex[:8],
            items=resolved,
            address=final_address,
            email=final_email,
            quote=quote,
            customer_turn_at_creation=self.session.customer_turns,
        )
        self.session.summaries[summary.id] = summary
        return {
            "summary_id": summary.id,
            "lines": [asdict(line) for line in quote.lines],
            "shipping_option": quote.shipping_option,
            "shipping_bdt": quote.shipping_bdt,
            "subtotal_bdt": quote.subtotal_bdt,
            "total_bdt": quote.total_bdt,
            "payment": "Cash on delivery",
            "address": asdict(final_address),
            "email": final_email,
            "next_step": "Read this summary to the customer and ask for a clear yes.",
        }

    def _order_contact(self, given: dict, email: str | None) -> tuple[Address, str]:
        customer = self.session.identity.customer
        base = customer.default_address if customer else None
        fields = {
            "first_name": given.get("first_name") or (base.first_name if base else ""),
            "last_name": given.get("last_name") or (base.last_name if base else ""),
            "address_1": given.get("address_1") or (base.address_1 if base else ""),
            "city": given.get("city") or (base.city if base else ""),
            "postal_code": given.get("postal_code") or (base.postal_code if base else ""),
            "phone": given.get("phone") or (base.phone if base else "")
            or (customer.phone if customer else ""),
        }
        final_email = email or (customer.email if customer else "")
        missing = [k for k in ("first_name", "address_1", "city", "phone") if not fields[k]]
        if not final_email:
            missing.append("email")
        if missing:
            raise ToolError(f"Ask the customer for: {', '.join(missing)}.")
        if not _is_bd_mobile(fields["phone"]):
            raise ToolError("The phone must be a Bangladesh mobile number, for example 01712345678.")
        return Address(**fields), final_email

    async def _tool_place_order(self, summary_id: str) -> dict:
        summary = self.session.summaries.get(summary_id)
        if summary is None:
            raise ToolError("Unknown summary_id. Call prepare_order first.")
        if self.session.customer_turns <= summary.customer_turn_at_creation:
            raise ToolError("Read the summary to the customer and wait for a clear yes.")
        order = await self.store.place_order(
            summary.items, summary.address, summary.email, self.session.identity.token
        )
        del self.session.summaries[summary_id]
        self.session.verified_orders.add(order.display_id)
        return {"order": _order_view(order)}
