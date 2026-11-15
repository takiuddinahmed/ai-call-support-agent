import pytest

from support_agent.store.client import StoreAuthError

pytestmark = pytest.mark.store


async def test_list_products_has_prices_stock_and_specs(store):
    products = await store.list_products()
    assert len(products) == 55
    by_handle = {p.handle: p for p in products}

    esp = by_handle["esp32-devkit-v1"]
    assert esp.variants[0].price_bdt == 550
    assert esp.variants[0].stock > 0
    assert esp.specs["wifi"] is True
    assert esp.category == "Microcontrollers and Boards"
    assert "esp32" in esp.tags

    pi = by_handle["raspberry-pi-5"]
    assert sorted(v.title for v in pi.variants) == ["2 GB", "4 GB", "8 GB"]
    assert {v.title: v.specs["ram_mb"] for v in pi.variants}["4 GB"] == 4096

    assert by_handle["stm32-blue-pill"].variants[0].stock == 0


async def test_get_product(store):
    assert (await store.get_product("esp32-devkit-v1")).title.startswith("ESP32 DevKit")
    assert await store.get_product("no-such-product") is None


async def test_login_customer_and_orders(store):
    token = await store.login("rahim@example.com", "Demo1234!")
    customer = await store.get_customer(token)
    assert customer.phone == "+8801711000001"
    assert customer.first_name == "Rahim"
    assert customer.default_address.city == "Dhaka"

    orders = {o.display_id: o for o in await store.list_orders(token)}
    assert {1, 6} <= set(orders)
    assert orders[6].tracking.status == "delivered"
    assert orders[6].items


async def test_wrong_password(store):
    with pytest.raises(StoreAuthError):
        await store.login("rahim@example.com", "wrong")


async def test_lookup_and_track(store):
    assert await store.lookup_order(6, "x@example.com", None) is None
    order = await store.lookup_order(6, "rahim@example.com", None)
    assert order.tracking.history[-1].label == "Delivered"
    assert (await store.track(order.tracking.tracking_number)).order_display_id == 6
    assert await store.track("DEMO-0000000000") is None
