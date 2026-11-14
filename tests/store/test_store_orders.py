import uuid

import pytest

from support_agent.store.client import StoreError
from support_agent.store.models import Address

pytestmark = pytest.mark.store


async def variant_id(store, handle):
    return (await store.get_product(handle)).variants[0].id


def address(city="Dhaka"):
    return Address("Bench", "Test", "Road 1, House 2", city, "1212", "01700000000")


async def test_quote_inside_dhaka(store):
    esp = await variant_id(store, "esp32-devkit-v1")
    quote = await store.quote([(esp, 2)], "Dhaka")
    assert quote.shipping_option == "Inside Dhaka"
    assert quote.shipping_bdt == 80
    assert quote.subtotal_bdt == 1100
    assert quote.total_bdt == 1180
    assert quote.lines[0].unit_price_bdt == 550
    assert quote.lines[0].line_total_bdt == 1100


async def test_quote_outside_dhaka_trims_city(store):
    esp = await variant_id(store, "esp32-devkit-v1")
    quote = await store.quote([(esp, 1)], " sylhet ")
    assert quote.shipping_option == "Outside Dhaka"
    assert quote.shipping_bdt == 150


async def test_quote_above_stock_names_the_stock(store):
    stm = await variant_id(store, "stm32-blue-pill")
    with pytest.raises(StoreError, match="0 in stock"):
        await store.quote([(stm, 1)], "Dhaka")


async def test_place_order_as_guest(store):
    email = f"bench-test-{uuid.uuid4().hex[:8]}@example.com"
    breadboard = await variant_id(store, "breadboard-830")
    order = await store.place_order([(breadboard, 1)], address("Chattogram"), email, None)
    assert order.status == "pending"
    assert order.shipping_option == "Outside Dhaka"
    found = await store.lookup_order(order.display_id, email, None)
    assert found.display_id == order.display_id


async def register_test_customer(store) -> tuple[str, str]:
    """Makes a new store account, so that tests do not add orders to the demo customers."""
    email = f"bench-customer-{uuid.uuid4().hex[:8]}@example.com"
    response = await store._http.post(
        "/auth/customer/emailpass/register", json={"email": email, "password": "Bench1234!"}
    )
    token = response.json()["token"]
    await store._request(
        "POST", "/store/customers", token,
        json={"email": email, "first_name": "Bench", "last_name": "Test", "phone": "01700000000"},
    )
    return await store.login(email, "Bench1234!"), email


async def test_place_order_as_customer_links_the_account(store):
    token, email = await register_test_customer(store)
    wires = await variant_id(store, "jumper-wires-120")
    order = await store.place_order([(wires, 1)], address(), email, token)
    assert order.display_id in [o.display_id for o in await store.list_orders(token)]
