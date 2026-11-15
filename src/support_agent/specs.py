"""Product search by text, category, and typed specification values."""

import re
from dataclasses import dataclass
from typing import Literal

from support_agent.store.models import Product

# Copy of the spec key list in demo-store/README.md (key to value type).
SPEC_KEYS: dict[str, str] = {
    "operating_voltage_min_v": "number",
    "operating_voltage_max_v": "number",
    "logic_level_v": "number",
    "current_ma": "number",
    "rated_voltage_v": "number",
    "max_current_a": "number",
    "capacity_mah": "number",
    "output_power_w": "number",
    "interface": "string[]",
    "wifi": "boolean",
    "bluetooth": "boolean",
    "wireless_protocol": "string",
    "wireless_range_m": "number",
    "cpu": "string",
    "clock_mhz": "number",
    "ram_mb": "number",
    "flash_mb": "number",
    "gpio_pins": "number",
    "measurement": "string",
    "range_min": "number",
    "range_max": "number",
    "range_unit": "string",
    "accuracy": "string",
    "torque_kg_cm": "number",
    "speed_rpm": "number",
    "step_angle_deg": "number",
    "channels": "number",
    "resolution": "string",
    "screen_size_in": "number",
    "dimensions_mm": "string",
    "weight_g": "number",
}

Op = Literal["eq", "gte", "lte", "contains"]


class SpecFilterError(ValueError):
    """A filter has an unknown key, an unknown operator, or a wrong value type."""


@dataclass
class SpecFilter:
    key: str
    op: Op
    value: str | float | bool


def _check(f: SpecFilter) -> None:
    kind = SPEC_KEYS.get(f.key)
    if kind is None:
        raise SpecFilterError(f"Unknown spec key '{f.key}'. Known keys: {', '.join(SPEC_KEYS)}.")
    if f.op not in ("eq", "gte", "lte", "contains"):
        raise SpecFilterError(f"Unknown operator '{f.op}'. Use eq, gte, lte, or contains.")
    if kind == "number":
        if isinstance(f.value, bool) or not isinstance(f.value, (int, float)):
            raise SpecFilterError(f"The value of '{f.key}' must be a number.")
        if f.op == "contains":
            raise SpecFilterError(f"Use eq, gte, or lte with '{f.key}'.")
    elif kind == "boolean":
        if not isinstance(f.value, bool) or f.op != "eq":
            raise SpecFilterError(f"Use eq with true or false for '{f.key}'.")
    elif f.op in ("gte", "lte"):
        raise SpecFilterError(f"Use eq or contains with '{f.key}'.")


def _match(value, f: SpecFilter) -> bool:
    if value is None:
        return False
    kind = SPEC_KEYS[f.key]
    if kind == "number":
        if f.op == "eq":
            return value == f.value
        return value >= f.value if f.op == "gte" else value <= f.value
    if kind == "boolean":
        return value is f.value
    wanted = str(f.value).lower()
    if kind == "string[]":
        items = [str(v).lower() for v in value]
        return wanted in items
    text = str(value).lower()
    return wanted == text if f.op == "eq" else wanted in text


def _matches_filter(product: Product, f: SpecFilter) -> bool:
    spec_sets = [product.specs] + [v.specs for v in product.variants]
    return any(_match(specs.get(f.key), f) for specs in spec_sets)


STOP_WORDS = {
    "a", "an", "the", "for", "of", "to", "with", "and", "or", "in", "on", "my", "i", "me",
    "need", "want", "some", "any", "is", "are", "this", "that", "it", "do", "you", "have",
    "can", "please", "buy", "which", "what", "good", "best", "suggest", "component", "components",
}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _query_words(text: str) -> set[str]:
    return _words(text) - STOP_WORDS


def _text_score(product: Product, words: set[str]) -> tuple[int, int]:
    """Returns (number of query words found, weighted score). Words match whole words only.
    A word in the title counts double."""
    title = _words(product.title)
    other = _words(" ".join([product.description, product.category or "", " ".join(product.tags)]))
    found = [w for w in words if w in title or w in other]
    return len(found), sum(2 if w in title else 1 for w in found)


def filter_products(
    products: list[Product],
    text: str | None,
    category: str | None,
    filters: list[SpecFilter],
    in_stock_only: bool = False,
) -> list[Product]:
    for f in filters:
        _check(f)
    words = _query_words(text) if text else set()
    result = []
    scores: dict[str, tuple[int, int]] = {}
    for product in products:
        if words:
            scores[product.id] = _text_score(product, words)
            if scores[product.id][0] == 0:
                continue
        if category and (product.category or "").lower() != category.strip().lower():
            continue
        if in_stock_only and product.total_stock <= 0:
            continue
        if all(_matches_filter(product, f) for f in filters):
            result.append(product)
    if words:
        # Products with all the words come first. Without such a product, any word counts,
        # because models often put several part names in one search.
        complete = [p for p in result if scores[p.id][0] == len(words)]
        result = complete or result
        result.sort(key=lambda p: -scores[p.id][1])
    return result
