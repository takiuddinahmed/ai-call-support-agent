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
    ok, _data = await run(customer(), store, "get_order_status", display_id=3)
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


async def test_empty_search_gives_a_retry_hint():
    ok, data = await run(guest(), FakeStore(), "search_products", text="submarine")
    assert ok and data["count"] == 0
    assert "fewer words" in data["hint"]


async def test_filters_that_match_nothing_fall_back_to_the_words():
    ok, data = await run(guest(), FakeStore(), "search_products", text="ESP32",
                         filters=[{"key": "ram_mb", "op": "gte", "value": 999999}])
    assert ok and [p["handle"] for p in data["products"]] == ["esp32-devkit-v1"]
    assert "No product matched the filters" in data["note"]


# Final review fixes

import httpx

from support_agent.store.client import StoreError


@pytest.mark.parametrize("name,args", [
    ("get_order_status", {"display_id": "six"}),
    ("prepare_order", {"items": "esp32"}),
    ("prepare_order", {"items": [ITEM], "address": "Dhaka, House 5"}),
    ("search_products", {"filters": ["wifi"]}),
    ("search_products", {"text": 5}),
    ("track_shipment", {"tracking_number": 12}),
])
async def test_bad_argument_shapes_give_tool_errors(name, args):
    result = await ToolRunner(FakeStore(), customer()).run(name, json.dumps(args))
    assert result.ok is False
    assert result.content


async def test_store_down_gives_tool_error():
    class DownStore(FakeStore):
        async def list_products(self):
            raise httpx.ConnectError("refused")

    result = await ToolRunner(DownStore(), guest()).run("search_products", '{"text": "esp32"}')
    assert result.ok is False and "store" in result.content.lower()


async def test_new_summary_replaces_old_ones():
    session = customer()
    store = FakeStore()
    _, first = await run(session, store, "prepare_order", items=[ITEM])
    _, second = await run(session, store, "prepare_order", items=[ITEM],
                          address=dict(GUEST_ADDRESS))
    session.customer_turns += 1
    ok, message = await run(session, store, "place_order", summary_id=first["summary_id"])
    assert not ok and "Unknown summary" in message
    ok, _ = await run(session, store, "place_order", summary_id=second["summary_id"])
    assert ok and store.placed[0][1].city == "Sylhet"


async def test_failed_place_order_cannot_be_retried_into_a_duplicate():
    class FailingStore(FakeStore):
        async def place_order(self, items, address, email, token):
            self.placed.append(items)
            raise StoreError("The lookup after the order failed.")

    session = customer()
    store = FailingStore()
    _, summary = await run(session, store, "prepare_order", items=[ITEM])
    session.customer_turns += 1
    ok, _ = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok
    ok, message = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok and "Unknown summary" in message
    assert len(store.placed) == 1


def voice_guest():
    return Session.new(Identity.guest(), "voice")


async def test_voice_place_order_needs_heard_summary():
    store = FakeStore()
    session = voice_guest()
    ok, summary = await run(session, store, "prepare_order", items=[ITEM], address=GUEST_ADDRESS)
    assert ok
    turn = session.customer_turns
    session.customer_turns += 1
    ok, message = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok and "Call prepare_order again with the same details" in message
    assert store.placed == []
    session.mark_summaries_heard(turn)
    ok, _ = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert ok and len(store.placed) == 1


async def test_voice_place_order_needs_a_turn_after_the_held_speech():
    store = FakeStore()
    session = voice_guest()
    ok, summary = await run(session, store, "prepare_order", items=[ITEM], address=GUEST_ADDRESS)
    assert ok
    turn = session.customer_turns
    # One caller text waited during the summary audio. It becomes turn + 1.
    session.mark_summaries_heard(turn, approvable_after_turn=turn + 1)
    session.customer_turns = turn + 1
    ok, message = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert not ok and "spoke before the summary ended" in message
    assert store.placed == []
    session.customer_turns = turn + 2
    ok, _ = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert ok and len(store.placed) == 1


async def test_chat_place_order_needs_no_heard_flag():
    store = FakeStore()
    session = customer()
    ok, summary = await run(session, store, "prepare_order", items=[ITEM])
    session.customer_turns += 1
    ok, _ = await run(session, store, "place_order", summary_id=summary["summary_id"])
    assert ok and len(store.placed) == 1


async def test_voice_guest_order_without_email_uses_phone_email():
    address = {**GUEST_ADDRESS, "phone": "01712345678"}
    ok, summary = await run(voice_guest(), FakeStore(), "prepare_order", items=[ITEM], address=address)
    assert ok and summary["email"] == "voice-8801712345678@guest.robopartsbd.local"


async def test_chat_guest_order_without_email_still_asks_for_email():
    ok, message = await run(guest(), FakeStore(), "prepare_order", items=[ITEM], address=GUEST_ADDRESS)
    assert not ok and "email" in message


async def test_account_without_an_address_gives_its_name_and_phone():
    from dataclasses import replace

    no_address = replace(RAHIM, default_address=None)
    session = Session.new(Identity(kind="customer", token="tok", customer=no_address), "voice")
    store = FakeStore()
    items = [{"handle": "esp32-devkit-v1", "quantity": 1}]
    ok, content = await run(session, store, "prepare_order", items=items)
    assert not ok
    # Only the address is missing. The account has the name and the phone.
    assert "address, city" in content
    assert "name" not in content and "phone" not in content
    assert "no saved address" in content
    ok, summary = await run(session, store, "prepare_order", items=items,
                            address={"address_1": "Road 5, Mirpur", "city": "Dhaka"})
    assert ok, summary
    assert summary["address"]["first_name"] == "Rahim"
    assert summary["address"]["last_name"] == "Uddin"
    assert summary["address"]["phone"] == "+8801711000001"
