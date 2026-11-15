"""The only code that calls the store HTTP API."""

import time

import httpx

from support_agent.config import Settings, resolve_publishable_key
from support_agent.store.models import (
    Address,
    Customer,
    OrderItem,
    OrderView,
    Product,
    Quote,
    QuoteLine,
    TrackingStep,
    TrackingView,
    Variant,
)

PRODUCT_FIELDS = (
    "*variants.calculated_price,+variants.inventory_quantity,+variants.metadata,"
    "+metadata,*categories,*tags"
)


class StoreError(Exception):
    """The store refused a request. The message is safe to show to the model."""


class StoreAuthError(StoreError):
    pass


ZONE_OPTION_NAMES = {"inside_dhaka": "Inside Dhaka", "outside_dhaka": "Outside Dhaka"}


def shipping_zone_for_city(city: str) -> str:
    """Same rule as the store: Dhaka (any case, trimmed) is inside, other cities outside."""
    return "inside_dhaka" if city.strip().lower() == "dhaka" else "outside_dhaka"


def _bdt(amount) -> int:
    return int(round(float(amount or 0)))


def _tracking(data: dict | None) -> TrackingView | None:
    if not data:
        return None
    return TrackingView(
        tracking_number=data["tracking_number"],
        status=data["status"],
        label=data["status_label"],
        order_display_id=data.get("order_display_id"),
        history=[TrackingStep(h["status"], h["status_label"], h["at"]) for h in data["history"]],
    )


def _order(data: dict) -> OrderView:
    return OrderView(
        display_id=data["display_id"],
        status=data["status"],
        created_at=data["created_at"],
        total_bdt=_bdt(data.get("total")),
        items=[
            OrderItem(i["title"], i.get("variant_title"), i["quantity"])
            for i in data.get("items", [])
        ],
        shipping_option=data.get("shipping_option"),
        tracking=_tracking(data.get("tracking")),
    )


def _address(data: dict) -> Address:
    return Address(
        first_name=data.get("first_name") or "",
        last_name=data.get("last_name") or "",
        address_1=data.get("address_1") or "",
        city=data.get("city") or "",
        postal_code=data.get("postal_code") or "",
        phone=data.get("phone") or "",
        country_code=data.get("country_code") or "bd",
    )


class StoreClient:
    def __init__(self, settings: Settings, http: httpx.AsyncClient | None = None):
        self.settings = settings
        self._http = http or httpx.AsyncClient(base_url=settings.store_url, timeout=30)
        self._key: str | None = None
        self._region_id: str | None = None
        self._products: list[Product] | None = None
        self._products_at = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    # Low-level helpers

    def _headers(self, token: str | None = None) -> dict:
        if self._key is None:
            self._key = resolve_publishable_key(self.settings)
        headers = {"x-publishable-api-key": self._key}
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    async def _request(self, method: str, path: str, token: str | None = None, **kwargs) -> dict:
        response = await self._http.request(method, path, headers=self._headers(token), **kwargs)
        if response.status_code >= 400:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            if response.status_code == 401:
                raise StoreAuthError(message)
            raise StoreError(message)
        return response.json()

    async def region_id(self) -> str:
        if self._region_id is None:
            data = await self._request("GET", "/store/regions")
            self._region_id = data["regions"][0]["id"]
        return self._region_id

    # Products

    async def list_products(self) -> list[Product]:
        fresh = time.monotonic() - self._products_at < self.settings.catalog_cache_seconds
        if self._products is not None and fresh:
            return self._products
        data = await self._request(
            "GET",
            "/store/products",
            params={"region_id": await self.region_id(), "fields": PRODUCT_FIELDS, "limit": 200},
        )
        self._products = [self._product(p) for p in data["products"]]
        self._products_at = time.monotonic()
        return self._products

    def _product(self, p: dict) -> Product:
        variants = [
            Variant(
                id=v["id"],
                title=v["title"],
                sku=v.get("sku"),
                price_bdt=_bdt((v.get("calculated_price") or {}).get("calculated_amount")),
                stock=int(v.get("inventory_quantity") or 0),
                specs=((v.get("metadata") or {}).get("specs") or {}),
            )
            for v in p.get("variants") or []
        ]
        categories = p.get("categories") or []
        return Product(
            id=p["id"],
            handle=p["handle"],
            title=p["title"],
            description=p.get("description") or "",
            category=categories[0]["name"] if categories else None,
            tags=[t["value"] for t in p.get("tags") or []],
            specs=((p.get("metadata") or {}).get("specs") or {}),
            variants=variants,
        )

    async def get_product(self, handle: str) -> Product | None:
        for product in await self.list_products():
            if product.handle == handle:
                return product
        return None

    # Customers

    async def login(self, email: str, password: str) -> str:
        response = await self._http.post(
            "/auth/customer/emailpass", json={"email": email, "password": password}
        )
        if response.status_code != 200:
            raise StoreAuthError("The email or the password is wrong.")
        return response.json()["token"]

    async def get_customer(self, token: str) -> Customer:
        data = (await self._request("GET", "/store/customers/me", token))["customer"]
        addresses = (await self._request("GET", "/store/customers/me/addresses", token))[
            "addresses"
        ]
        default = next((a for a in addresses if a.get("is_default_shipping")), None)
        default = default or (addresses[0] if addresses else None)
        return Customer(
            id=data["id"],
            email=data["email"],
            first_name=data.get("first_name") or "",
            last_name=data.get("last_name") or "",
            phone=data.get("phone"),
            default_address=_address(default) if default else None,
        )

    # Orders and tracking

    async def lookup_order(
        self, display_id: int, email: str | None, phone: str | None
    ) -> OrderView | None:
        params = {"display_id": str(display_id)}
        if email:
            params["email"] = email
        if phone:
            params["phone"] = phone
        response = await self._http.get(
            "/store/order-lookup", params=params, headers=self._headers()
        )
        if response.status_code in (400, 404):
            return None
        response.raise_for_status()
        return _order(response.json()["order"])

    async def list_orders(self, token: str) -> list[OrderView]:
        customer = await self.get_customer(token)
        data = await self._request(
            "GET", "/store/orders", token, params={"fields": "display_id", "limit": 50}
        )
        orders = []
        for summary in data["orders"]:
            order = await self.lookup_order(summary["display_id"], customer.email, None)
            if order:
                orders.append(order)
        return orders

    async def track(self, tracking_number: str) -> TrackingView | None:
        response = await self._http.get(
            f"/store/tracking/{tracking_number}", headers=self._headers()
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return _tracking(response.json()["tracking"])

    # Quotes and orders

    async def _variant_index(self) -> dict[str, tuple[Product, Variant]]:
        return {v.id: (p, v) for p in await self.list_products() for v in p.variants}

    async def _shipping_option(self, cart_id: str, city: str) -> tuple[str, str, int]:
        """Returns (option ID, name, fee) of the delivery option for the city."""
        zone = shipping_zone_for_city(city)
        data = await self._request("GET", "/store/shipping-options", params={"cart_id": cart_id})
        for option in data["shipping_options"]:
            if (option.get("data") or {}).get("zone") == zone:
                return option["id"], option["name"], _bdt(option.get("amount"))
        raise StoreError(f"The store has no delivery option for {city}.")

    async def _new_cart(self, city: str, token: str | None = None, **extra) -> str:
        body = {
            "region_id": await self.region_id(),
            "shipping_address": {"city": city.strip(), "country_code": "bd"},
            **extra,
        }
        return (await self._request("POST", "/store/carts", token, json=body))["cart"]["id"]

    async def quote(self, items: list[tuple[str, int]], city: str) -> Quote:
        index = await self._variant_index()
        lines = []
        for variant_id, quantity in items:
            if variant_id not in index:
                raise StoreError(f"No product variant has the ID {variant_id}.")
            product, variant = index[variant_id]
            if quantity > variant.stock:
                raise StoreError(
                    f"{product.title} ({variant.title}) has only {variant.stock} in stock."
                )
            lines.append(
                QuoteLine(
                    variant_id=variant_id,
                    product_title=product.title,
                    variant_title=variant.title,
                    quantity=quantity,
                    unit_price_bdt=variant.price_bdt,
                    line_total_bdt=variant.price_bdt * quantity,
                )
            )
        cart_id = await self._new_cart(city)
        _, option_name, fee = await self._shipping_option(cart_id, city)
        subtotal = sum(line.line_total_bdt for line in lines)
        return Quote(lines, option_name, fee, subtotal, subtotal + fee)

    async def place_order(
        self, items: list[tuple[str, int]], address: Address, email: str, token: str | None
    ) -> OrderView:
        shipping_address = {
            "first_name": address.first_name,
            "last_name": address.last_name,
            "address_1": address.address_1,
            "city": address.city.strip(),
            "postal_code": address.postal_code,
            "phone": address.phone,
            "country_code": address.country_code,
        }
        cart_id = (
            await self._request(
                "POST",
                "/store/carts",
                token,
                json={
                    "region_id": await self.region_id(),
                    "email": email,
                    "shipping_address": shipping_address,
                    "billing_address": shipping_address,
                },
            )
        )["cart"]["id"]
        for variant_id, quantity in items:
            await self._request(
                "POST",
                f"/store/carts/{cart_id}/line-items",
                token,
                json={"variant_id": variant_id, "quantity": quantity},
            )
        option_id, _, _ = await self._shipping_option(cart_id, address.city)
        await self._request(
            "POST", f"/store/carts/{cart_id}/shipping-methods", token, json={"option_id": option_id}
        )
        collection = (
            await self._request("POST", "/store/payment-collections", token, json={"cart_id": cart_id})
        )["payment_collection"]
        await self._request(
            "POST",
            f"/store/payment-collections/{collection['id']}/payment-sessions",
            token,
            json={"provider_id": "pp_system_default"},
        )
        result = await self._request("POST", f"/store/carts/{cart_id}/complete", token)
        if result.get("type") != "order":
            raise StoreError((result.get("error") or {}).get("message", "The order failed."))
        self._products = None  # stock changed
        placed = result["order"]
        try:
            order = await self.lookup_order(placed["display_id"], email, None)
        except httpx.HTTPError:
            order = None
        # The order exists now. If the lookup fails, return what the store sent back.
        return order or OrderView(
            display_id=placed["display_id"],
            status=placed.get("status", "pending"),
            created_at=placed.get("created_at", ""),
            total_bdt=_bdt(placed.get("total")),
            items=[],
            shipping_option=None,
            tracking=None,
        )
