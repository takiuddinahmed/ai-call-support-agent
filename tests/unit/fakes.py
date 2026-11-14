"""A fake store with the same method names as StoreClient, for unit tests."""

from support_agent.store.client import StoreError, shipping_zone_for_city
from support_agent.store.models import (
    Address, Customer, OrderItem, OrderView, Product, Quote, QuoteLine,
    TrackingStep, TrackingView, Variant,
)

ESP = Product("p1", "esp32-devkit-v1", "ESP32 DevKit V1", "Wi-Fi board", "Microcontrollers and Boards",
              ["esp32"], {"wifi": True}, [Variant("v1", "Default", "ESP", 550, 120, {})])
PI = Product("p2", "raspberry-pi-5", "Raspberry Pi 5", "SBC", "Single-Board Computers", [],
             {"cpu": "BCM2712"},
             [Variant("v2a", "2 GB", None, 8500, 12, {"ram_mb": 2048}),
              Variant("v2b", "4 GB", None, 10500, 20, {"ram_mb": 4096})])
STM = Product("p3", "stm32-blue-pill", "STM32 Blue Pill", "ARM board", "Microcontrollers and Boards",
              [], {}, [Variant("v3", "Default", None, 420, 0, {})])

RAHIM_ADDRESS = Address("Rahim", "Uddin", "House 12, Dhanmondi", "Dhaka", "1205", "+8801711000001")
RAHIM = Customer("cus_1", "rahim@example.com", "Rahim", "Uddin", "+8801711000001", RAHIM_ADDRESS)

TRACKING = TrackingView("DEMO-1234567890", "in_transit", "In transit", 3,
                        [TrackingStep("picked_up", "Picked up", "2026-11-14T10:00:00Z"),
                         TrackingStep("in_transit", "In transit", "2026-11-14T16:00:00Z")])
ORDER_3 = OrderView(3, "pending", "2026-11-14T09:00:00Z", 1180,
                    [OrderItem("ESP32 DevKit V1", "Default", 2)], "Inside Dhaka", TRACKING)


class FakeStore:
    def __init__(self):
        self.calls: list[tuple] = []
        self.placed: list[tuple] = []

    async def list_products(self):
        return [ESP, PI, STM]

    async def get_product(self, handle):
        return next((p for p in [ESP, PI, STM] if p.handle == handle), None)

    async def list_orders(self, token):
        self.calls.append(("list_orders", token))
        return [ORDER_3]

    async def lookup_order(self, display_id, email, phone):
        self.calls.append(("lookup_order", display_id, email, phone))
        if display_id == 3 and (email == "rahim@example.com" or phone in ("+8801711000001", "01711000001")):
            return ORDER_3
        return None

    async def track(self, tracking_number):
        self.calls.append(("track", tracking_number))
        return TRACKING if tracking_number == TRACKING.tracking_number else None

    async def quote(self, items, city):
        index = {v.id: (p, v) for p in [ESP, PI, STM] for v in p.variants}
        lines = []
        for variant_id, quantity in items:
            product, variant = index[variant_id]
            if quantity > variant.stock:
                raise StoreError(f"{product.title} ({variant.title}) has only {variant.stock} in stock.")
            lines.append(QuoteLine(variant_id, product.title, variant.title, quantity,
                                   variant.price_bdt, variant.price_bdt * quantity))
        inside = shipping_zone_for_city(city) == "inside_dhaka"
        fee = 80 if inside else 150
        subtotal = sum(line.line_total_bdt for line in lines)
        return Quote(lines, "Inside Dhaka" if inside else "Outside Dhaka", fee, subtotal, subtotal + fee)

    async def place_order(self, items, address, email, token):
        self.placed.append((items, address, email, token))
        return OrderView(10, "pending", "2026-11-15T10:00:00Z", 1180,
                         [OrderItem("ESP32 DevKit V1", "Default", 2)], "Inside Dhaka", None)


class FakeLlm:
    """Plays scripted rounds. Each round is a text answer or a list of (name, arguments)."""

    def __init__(self, rounds, error: Exception | None = None):
        self.rounds = list(rounds)
        self.error = error
        self.seen: list[list[dict]] = []

    async def stream(self, model, messages, tools):
        from support_agent.llm import LlmChunk, ToolCallDelta

        self.seen.append(list(messages))
        if self.error:
            raise self.error
        step = self.rounds.pop(0)
        if isinstance(step, str):
            for word in step.split(" "):
                yield LlmChunk(text=word + " ")
            yield LlmChunk(usage={"completion_tokens": len(step.split())})
        else:
            yield LlmChunk(tool_calls=[
                ToolCallDelta(index=i, id=f"call_{i}", name=name, arguments=args)
                for i, (name, args) in enumerate(step)
            ])


async def _fake_login(self, email, password):
    from support_agent.store.client import StoreAuthError

    if (email, password) != ("rahim@example.com", "Demo1234!"):
        raise StoreAuthError("The email or the password is wrong.")
    return "tok"


async def _fake_get_customer(self, token):
    return RAHIM


FakeStore.login = _fake_login
FakeStore.get_customer = _fake_get_customer


class FakeModels:
    async def list_models(self):
        return ["qwen3.5:4b", "other:7b"]
