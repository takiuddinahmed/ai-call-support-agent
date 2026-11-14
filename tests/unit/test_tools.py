import json

import pytest

from support_agent.session import Identity, Session
from support_agent.tools import TOOL_SPECS, ToolRunner
from tests.unit.fakes import RAHIM, FakeStore


def guest():
    return Session.new(Identity.guest(), "chat")


def customer():
    return Session.new(Identity(kind="customer", token="tok", customer=RAHIM), "chat")


async def run(session, store, name, **args):
    result = await ToolRunner(store, session).run(name, json.dumps(args))
    return result.ok, (json.loads(result.content) if result.ok else result.content)


def test_tool_specs_names():
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert names == ["search_products", "get_product", "list_my_orders", "get_order_status",
                     "track_shipment", "prepare_order", "place_order"]


async def test_search_with_filter_and_limit():
    ok, data = await run(guest(), FakeStore(), "search_products",
                         filters=[{"key": "wifi", "op": "eq", "value": True}])
    assert ok and [p["handle"] for p in data["products"]] == ["esp32-devkit-v1"]
    assert data["products"][0]["price_bdt"] == 550


async def test_search_unknown_key_is_tool_error():
    ok, message = await run(guest(), FakeStore(), "search_products",
                            filters=[{"key": "voltage", "op": "gte", "value": 5}])
    assert not ok and "Unknown spec key" in message


async def test_get_product_unknown_handle():
    ok, message = await run(guest(), FakeStore(), "get_product", handle="nope")
    assert not ok and "No product has the handle" in message


async def test_bad_json_and_unknown_tool():
    runner = ToolRunner(FakeStore(), guest())
    assert not (await runner.run("search_products", "{bad")).ok
    assert not (await runner.run("delete_everything", "{}")).ok


async def test_list_my_orders_needs_login():
    store = FakeStore()
    ok, message = await run(guest(), store, "list_my_orders")
    assert not ok and "not logged in" in message
    assert store.calls == []
    ok, data = await run(customer(), store, "list_my_orders")
    assert ok and data["orders"][0]["display_id"] == 3


async def test_guest_order_status_needs_proof():
    store = FakeStore()
    session = guest()
    ok, message = await run(session, store, "get_order_status", display_id=3)
    assert not ok and "email or phone" in message
    assert store.calls == []
    ok, message = await run(session, store, "get_order_status", display_id=3, email="x@example.com")
    assert not ok and "No order matches" in message
    ok, data = await run(session, store, "get_order_status", display_id=3, phone="01711000001")
    assert ok and data["order"]["status"] == "pending"
    assert 3 in session.verified_orders


async def test_customer_order_status_uses_account_email():
    store = FakeStore()
    ok, data = await run(customer(), store, "get_order_status", display_id=3)
    assert ok and store.calls[-1] == ("lookup_order", 3, "rahim@example.com", None)


async def test_track_shipment_gives_status_only():
    ok, data = await run(guest(), FakeStore(), "track_shipment", tracking_number="demo-1234567890")
    assert ok and data["status"] == "In transit"
    assert "items" not in data


ITEM = {"handle": "esp32-devkit-v1", "quantity": 2}
GUEST_ADDRESS = {"first_name": "Karim", "last_name": "Ali", "address_1": "Road 3",
                 "city": "Sylhet", "postal_code": "3100", "phone": "01811111111"}


async def test_prepare_then_place_needs_a_new_customer_turn():
    store = FakeStore()
    session = customer()
    ok, summary = await run(session, store, "prepare_order", items=[ITEM])
    assert ok and summary["total_bdt"] == 1180 and summary["shipping_option"] == "Inside Dhaka"
    assert summary["address"]["city"] == "Dhaka"

    ok, message = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok and "wait for a clear yes" in message
    assert store.placed == []

    session.customer_turns += 1
    ok, data = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert ok and data["order"]["display_id"] == 10
    assert len(store.placed) == 1 and store.placed[0][3] == "tok"

    ok, message = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok and "Unknown summary" in message


async def test_prepare_order_guest_needs_address_and_email():
    ok, message = await run(guest(), FakeStore(), "prepare_order", items=[ITEM])
    assert not ok and "email" in message
    ok, summary = await run(guest(), FakeStore(), "prepare_order", items=[ITEM],
                            address=GUEST_ADDRESS, email="karim@example.com")
    assert ok and summary["shipping_option"] == "Outside Dhaka" and summary["shipping_bdt"] == 150


@pytest.mark.parametrize("quantity", [0, 21, 1000])
async def test_prepare_order_quantity_limits(quantity):
    ok, message = await run(customer(), FakeStore(), "prepare_order",
                            items=[{"handle": "esp32-devkit-v1", "quantity": quantity}])
    assert not ok and "1 to 20" in message


async def test_prepare_order_out_of_stock_and_variant_choice():
    ok, message = await run(customer(), FakeStore(), "prepare_order",
                            items=[{"handle": "stm32-blue-pill", "quantity": 1}])
    assert not ok and "0 in stock" in message
    ok, message = await run(customer(), FakeStore(), "prepare_order",
                            items=[{"handle": "raspberry-pi-5", "quantity": 1}])
    assert not ok and "2 GB" in message and "4 GB" in message
    ok, summary = await run(customer(), FakeStore(), "prepare_order",
                            items=[{"handle": "raspberry-pi-5", "variant_title": "4 gb", "quantity": 1}])
    assert ok and summary["lines"][0]["unit_price_bdt"] == 10500


async def test_prepare_order_rejects_bad_phone():
    bad = dict(GUEST_ADDRESS, phone="12345")
    ok, message = await run(guest(), FakeStore(), "prepare_order", items=[ITEM],
                            address=bad, email="karim@example.com")
    assert not ok and "mobile number" in message
