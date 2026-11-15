import pytest

from support_agent.specs import SpecFilter, SpecFilterError, filter_products
from support_agent.store.models import Product, Variant


def product(handle, specs, stock=10, title=None, variants=None, description="", tags=()):
    return Product(
        id=handle,
        handle=handle,
        title=title or handle,
        description=description,
        category="Sensors",
        tags=list(tags),
        specs=specs,
        variants=variants or [Variant(handle + "-v", "Default", None, 100, stock, {})],
    )


SENSOR_5V = product("sr04", {"operating_voltage_max_v": 5, "interface": ["GPIO"]},
                    title="HC-SR04 Ultrasonic Distance Sensor")
SENSOR_3V = product("vl53", {"operating_voltage_max_v": 3.3, "interface": ["I2C", "SPI"]},
                    title="VL53L0X Time-of-Flight Distance Sensor")
ESP = product("esp32", {"wifi": True, "cpu": "Xtensa LX6"}, title="ESP32 DevKit")
PI = product(
    "pi5",
    {"cpu": "BCM2712"},
    title="Raspberry Pi 5",
    variants=[
        Variant("a", "2 GB", None, 8500, 3, {"ram_mb": 2048}),
        Variant("b", "4 GB", None, 10500, 0, {"ram_mb": 4096}),
    ],
)
EMPTY = product("stm32", {"wifi": False}, stock=0, title="STM32 Blue Pill")
ALL = [SENSOR_5V, SENSOR_3V, ESP, PI, EMPTY]


def handles(products):
    return [p.handle for p in products]


def test_gte_on_number():
    result = filter_products(ALL, None, None, [SpecFilter("operating_voltage_max_v", "gte", 5)])
    assert handles(result) == ["sr04"]


def test_contains_on_list_ignores_case():
    result = filter_products(ALL, None, None, [SpecFilter("interface", "contains", "i2c")])
    assert handles(result) == ["vl53"]


def test_eq_on_boolean():
    assert handles(filter_products(ALL, None, None, [SpecFilter("wifi", "eq", True)])) == ["esp32"]


def test_variant_specs_count():
    assert handles(filter_products(ALL, None, None, [SpecFilter("ram_mb", "gte", 4096)])) == ["pi5"]


def test_text_matches_any_word_ranked_by_matches():
    assert handles(filter_products(ALL, "distance", None, [])) == ["sr04", "vl53"]
    # sr04 matches both words, vl53 only one, so sr04 comes first.
    assert handles(filter_products(ALL, "ultrasonic distance", None, [])) == ["sr04", "vl53"]
    # No product has both words, but each word finds a product.
    assert handles(filter_products(ALL, "ESP32 Raspberry", None, [])) == ["esp32", "pi5"]
    assert filter_products(ALL, "submarine", None, []) == []


def test_unknown_key_raises():
    with pytest.raises(SpecFilterError):
        filter_products(ALL, None, None, [SpecFilter("voltage", "gte", 5)])


def test_wrong_value_type_raises():
    with pytest.raises(SpecFilterError):
        filter_products(ALL, None, None, [SpecFilter("operating_voltage_max_v", "gte", "high")])


def test_in_stock_only_drops_empty_products():
    assert "stm32" not in handles(filter_products(ALL, None, None, [], in_stock_only=True))
    assert "pi5" in handles(filter_products(ALL, None, None, [], in_stock_only=True))


def test_category_filter_ignores_case():
    assert len(filter_products(ALL, None, "sensors", [])) == 5
    assert filter_products(ALL, None, "Displays", []) == []
