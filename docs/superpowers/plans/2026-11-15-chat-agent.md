# Chat Support Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local text chat support agent for the demo store. It has a debug panel and a benchmark, and it is the shared core for the later voice agent.

**Architecture:** A Python package `support_agent` holds the core: configuration, store adapter, spec filter, tools, instructions, session, and agent loop. A FastAPI app in the same package serves the chat page and streams agent events as server-sent events. The model runs in Ollama on macOS and the agent calls it through the OpenAI-compatible API.

**Tech Stack:** Python 3.13 with `uv`, FastAPI, uvicorn, and httpx. The `openai` Python SDK is the OpenAI-compatible client. `pydantic-settings` reads the configuration, and PyYAML reads the test cases. Tests use pytest, pytest-asyncio, and ruff. Ollama 0.40.16 serves the models.

**Spec:** `agent/docs/superpowers/specs/2026-11-15-chat-agent-design.md`

## Global Constraints

- All AI runs locally. No cloud AI service. Model server: Ollama at `http://localhost:11434/v1`.
- AI models together use at most 12 GB of memory. `qwen3.8:27b-mlx` is a quality reference only.
- The agent talks to the store only through the store HTTP API at `http://localhost:9000`. No store code changes, except the compose `name: ai-audio-call` line.
- The agent never states a product, price, stock number, or order status that the store API did not return.
- `place_order` runs only with a summary ID that the session holds, and only after a customer message that came after that summary.
- An order is shown only after the session made sure that it belongs to this person. A tracking number alone gives the shipment status and history only.
- Channel setting: `chat` or `voice`. Voice style: one to three short sentences, no lists or tables, numbers and prices in words.
- Answer in the customer's language: Bangla, English, or mixed. Product names and spec values stay in English.
- Delivery option follows the city: "Inside Dhaka" for Dhaka (trimmed, case-insensitive), "Outside Dhaka" otherwise.
- Unsupported requests (cancel, return, refund, address change after order): the agent says it cannot do them and tells the customer to contact the store team.
- Success numbers: at least 90 percent of test conversations pass. Time to first answer token under 1 second for most turns without a tool call. Chosen model under 12 GB.
- All user-facing text and documents follow the user's Simplified Technical English rules.

## Review Focus

1. The customer answers the order summary with "no", "wait", or a change ("send it to Sylhet"). Expect no order, and a new summary after a change. Task 6 tests the code gate. Task 11 tests the model with Bangla and English cases.
2. The model calls a tool with bad arguments (unknown product handle, quantity 0 or 1000, unknown spec key). Expect a tool error message that the model can read, not a crash. Task 5 tests this.
3. A guest asks for "my orders" or an order by number without email or phone. Expect the agent to ask for the email or phone, and to show nothing. Task 5 tests this.
4. The model server is down or the model name is wrong. Expect an error event in the chat, not a hung page. Task 7 tests this.
5. A product with zero stock, or a quantity above stock. Expect `prepare_order` to refuse with the available stock. Task 5 tests this.

---

## File Structure

```
ai-audio-call/
  demo-store/                      the store repository (moved, Task 1)
  agent/                           the agent repository (new, Task 1)
    pyproject.toml
    .env.example                   configuration with demo values
    README.md
    src/support_agent/
      config.py                    Settings (pydantic-settings)
      store/client.py              StoreClient: all store HTTP calls
      store/models.py              Product, Variant, OrderView, TrackingView, Address dataclasses
      specs.py                     SpecFilter, filter_products()
      session.py                   Session, Identity, OrderSummary
      tools.py                     TOOL_SPECS, ToolRunner, ToolError
      instructions.py              build_instructions()
      llm.py                       LlmClient protocol, OpenAiCompatClient
      loop.py                      run_turn(), AgentEvent
      server/app.py                FastAPI app: login, chat SSE, models, reset
      server/static/index.html     chat page and debug panel
      bench/cases/*.yaml           scripted conversations
      bench/runner.py              benchmark command
    tests/
      unit/                        no network
      store/                       live demo store at localhost:9000 (marker: store)
      llm/                         live Ollama (marker: llm)
```

---

### Task 1: Split the workspace into two repositories

**Files:**
- Move: every file of the current repository root, including `.git`, into `ai-audio-call/demo-store/`, except the untracked `agent/` folder.
- Modify: `demo-store/docker-compose.yml` (first line `name: ai-audio-call`)
- Modify: `demo-store/README.md` (start commands run from `demo-store/`)
- Create: `agent/` git repository with `pyproject.toml`, `.gitignore`, `.env.example`, `src/support_agent/__init__.py`, `src/support_agent/config.py`, `tests/unit/test_config.py`

**Interfaces:**
- Produces: `Settings` with fields `store_url: str = "http://localhost:9000"`, `store_publishable_key: str = ""`, `store_key_file_command: str = "docker compose -f ../demo-store/docker-compose.yml exec -T medusa cat /shared/publishable_key"`, `llm_base_url: str = "http://localhost:11434/v1"`, `llm_model: str = "qwen3.5:4b"`, `llm_api_key: str = "ollama"`, `max_tool_rounds: int = 6`, `catalog_cache_seconds: int = 300`. `get_settings() -> Settings` reads `.env`. `resolve_publishable_key(settings) -> str` returns the setting. When the setting is empty, it runs the command.
- Produces: pytest markers `store` and `llm`. `uv run pytest -m "not store and not llm"` runs the unit tests.

- [ ] **Step 1: Count the store orders before the move.** Run `docker compose exec -T postgres psql -U medusa -tAc 'select count(*) from "order"'`. Write down the number.
- [ ] **Step 2: Run `docker compose down` (no `-v`), move the files into `demo-store/`, and add `name: ai-audio-call` to the compose file.**
- [ ] **Step 3: Run `cd demo-store && docker compose up -d`, wait for `localhost:9000/health`, and count the orders again.** Expected: the same number, and `docker volume ls` shows `ai-audio-call_pgdata` only once.
- [ ] **Step 4: Commit in `demo-store`.** `git commit -m "chore: move the store into demo-store and pin the compose project name"`
- [ ] **Step 5: Create the agent project with `uv init --package --name support-agent agent`, add the dependencies, and write `tests/unit/test_config.py`.** Tests: default values above. An env value `LLM_MODEL=gemma` overrides `llm_model`. `resolve_publishable_key` returns a set value without a command.
- [ ] **Step 6: Run `uv run pytest -m "not store and not llm"`.** Expected: FAIL before `config.py` exists, PASS after.
- [ ] **Step 7: Commit the design, this plan, and the scaffold in `agent`.** `git commit -m "chore: start the support agent project"`

---

### Task 2: Store client (reads and login)

**Files:**
- Create: `src/support_agent/store/models.py`, `src/support_agent/store/client.py`
- Test: `tests/store/test_store_reads.py`

**Interfaces:**
- Consumes: `Settings`, `resolve_publishable_key`.
- Produces dataclasses: `Variant(id, title, sku, price_bdt: int, stock: int, specs: dict)`, `Product(id, handle, title, description, category: str | None, tags: list[str], specs: dict, variants: list[Variant])`, `Address(first_name, last_name, address_1, city, postal_code, phone, country_code="bd")`, `OrderItem(title, variant_title, quantity)`, `TrackingStep(status, label, at)`, `TrackingView(tracking_number, status, label, order_display_id, history: list[TrackingStep])`, `OrderView(display_id: int, status, created_at, total_bdt: int, items: list[OrderItem], shipping_option: str | None, tracking: TrackingView | None)`, `Customer(id, email, first_name, last_name, phone, default_address: Address | None)`.
- Produces `StoreClient(settings, http: httpx.AsyncClient | None = None)` with async methods:
  - `list_products() -> list[Product]` (cached for `catalog_cache_seconds`, includes metadata, variants, BDT prices, stock)
  - `get_product(handle: str) -> Product | None`
  - `login(email: str, password: str) -> str` (customer token. Raises `StoreAuthError`)
  - `get_customer(token: str) -> Customer`
  - `list_orders(token: str) -> list[OrderView]` (tracking filled through `lookup_order` with the customer email)
  - `lookup_order(display_id: int, email: str | None, phone: str | None) -> OrderView | None` (`None` for 404)
  - `track(tracking_number: str) -> TrackingView | None`
- Store routes: `POST /auth/customer/emailpass`, `GET /store/customers/me`, `GET /store/customers/me/addresses`, `GET /store/orders`, `GET /store/order-lookup`, `GET /store/tracking/:n`, `GET /store/products?region_id=...&fields=*variants.calculated_price,+variants.inventory_quantity,+metadata,*categories,*tags&limit=100`, `GET /store/regions`.

- [ ] **Step 1: Write failing live tests (marker `store`).** Assert: `list_products()` returns 55 products. `esp32-devkit-v1` has `price_bdt == 550`, `stock` above 0, `specs["wifi"] is True`. `raspberry-pi-5` has variant titles `["2 GB", "4 GB", "8 GB"]` with `specs["ram_mb"]` set per variant. `stm32-blue-pill` has stock 0. `login("rahim@example.com", "Demo1234!")` returns a token. `get_customer(token).phone == "+8801711000001"` and its default address city is `"Dhaka"`. `list_orders(token)` contains display IDs 1 and 6, and order 6 has tracking status `"delivered"`. `lookup_order(6, "x@example.com", None) is None`. `track("DEMO-0000000000") is None`. A wrong password raises `StoreAuthError`.
- [ ] **Step 2: Run `uv run pytest -m store tests/store/test_store_reads.py`.** Expected: FAIL.
- [ ] **Step 3: Implement the models and the client.**
- [ ] **Step 4: Run the tests again.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add store client for products, login, orders, and tracking"`

---

### Task 3: Spec filter

**Files:**
- Create: `src/support_agent/specs.py`
- Test: `tests/unit/test_specs.py`

**Interfaces:**
- Consumes: `Product`, `Variant`.
- Produces: `SpecFilter(key: str, op: Literal["eq", "gte", "lte", "contains"], value: str | float | bool)`. `SPEC_KEYS: dict[str, str]` (key to type: `"number" | "boolean" | "string" | "string[]"`, copied from the store README table). `filter_products(products: list[Product], text: str | None, category: str | None, filters: list[SpecFilter], in_stock_only: bool = False) -> list[Product]`. `SpecFilterError` for an unknown key or a wrong value type.
- Rules: for a match, the product specs or the specs of one variant must match the filter. `text` matches title, description, tags, and category, case-insensitive, all words required. `contains` works on `string[]` (case-insensitive) and `string` (substring).

- [ ] **Step 1: Write failing tests with small hand-made products.** Assert: `gte operating_voltage_max_v 5` keeps a 5 V sensor and drops a 3.3 V one. `contains interface "i2c"` matches `["I2C","SPI"]`. `eq wifi True` keeps the ESP32. `gte ram_mb 4096` keeps a product whose 4 GB variant has it. Text `"distance sensor"` needs both words. Unknown key `voltage` raises `SpecFilterError`. `in_stock_only` drops a product with all variants at stock 0.
- [ ] **Step 2: Run `uv run pytest tests/unit/test_specs.py`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add product filter on typed spec values"`

---

### Task 4: Store client (orders)

**Files:**
- Modify: `src/support_agent/store/client.py`
- Test: `tests/store/test_store_orders.py`

**Interfaces:**
- Produces on `StoreClient`:
  - `shipping_zone_for_city(city: str) -> Literal["inside_dhaka", "outside_dhaka"]`
  - `quote(items: list[tuple[str, int]], city: str) -> Quote` where items are `(variant_id, quantity)`. `Quote(lines: list[QuoteLine], shipping_option: str, shipping_bdt: int, subtotal_bdt: int, total_bdt: int)`, `QuoteLine(variant_id, product_title, variant_title, quantity, unit_price_bdt, line_total_bdt)`. It reads prices and stock from `list_products()` and the delivery fee from the store shipping options of a temporary cart. For a quantity above stock, it raises `StoreError` with the available stock.
  - `place_order(items: list[tuple[str, int]], address: Address, email: str, token: str | None) -> OrderView` runs the store cart flow. It creates the cart and adds the items. For a logged-in customer, it sends the token. It sets the address and the email, and adds the matching shipping method. Then it creates the payment collection, starts `pp_system_default`, and completes the cart. Raises `StoreError` with the store message on failure.

- [ ] **Step 1: Write failing live tests (marker `store`).** Assert: `quote([(esp32_variant, 2)], "Dhaka")` gives `shipping_option == "Inside Dhaka"`, `shipping_bdt == 80`, `subtotal_bdt == 1100`, `total_bdt == 1180`. City `" sylhet "` gives `"Outside Dhaka"` and 150. Quantity above stock raises `StoreError` that names the stock. `place_order` for a guest with email `bench-test-<uuid>@example.com` returns an `OrderView` with `status == "pending"`, and `lookup_order` finds it with that email. `place_order` with Rahim's token makes an order that `list_orders(token)` contains.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add order quote and order placement to store client"`

---

### Task 5: Session and tools

**Files:**
- Create: `src/support_agent/session.py`, `src/support_agent/tools.py`
- Test: `tests/unit/test_tools.py` (uses a `FakeStore` in `tests/unit/fakes.py` with the same method names as `StoreClient`)

**Interfaces:**
- Consumes: `StoreClient` methods, `filter_products`, `SpecFilter`.
- Produces `Identity(kind: Literal["customer", "guest"], token: str | None, customer: Customer | None)`. `OrderSummary(id: str, items: list[tuple[str, int]], address: Address, email: str, quote: Quote, customer_turn_at_creation: int)`. `Session(id, identity, channel: Literal["chat","voice"], messages: list[dict], customer_turns: int, summaries: dict[str, OrderSummary], verified_orders: set[int])`.
- Produces `TOOL_SPECS: list[dict]` (OpenAI tool format) for `search_products`, `get_product`, `list_my_orders`, `get_order_status`, `track_shipment`, `prepare_order`, `place_order`.
- Produces `ToolRunner(store, session)` with `async run(name: str, arguments: str) -> ToolResult` where `ToolResult(ok: bool, content: str, ms: float)`. Bad JSON, unknown tool, `SpecFilterError`, `StoreError`, and rule failures give `ok=False` with a short English message for the model. Content is compact JSON.
- Rules:
  - `search_products(text?, category?, filters?, in_stock_only?)` returns at most 8 products with handle, title, lowest price, total stock, and the specs.
  - `list_my_orders()` for a guest returns `ok=False`, "The customer is not logged in. Ask for the order number and the email or phone."
  - `get_order_status(display_id, email?, phone?)`: a logged-in customer needs no email or phone (the tool uses the account email). A guest without email and phone gets `ok=False`. On success, add the ID to `verified_orders`.
  - `track_shipment(tracking_number)` returns status and history only.
  - `prepare_order(items: [{handle, variant_title?, quantity}], address?, email?)`: quantity 1 to 20. The logged-in customer default address and email fill missing fields. A guest needs name, address_1, city, phone, and email. Stores an `OrderSummary` with `customer_turn_at_creation = session.customer_turns` and returns the summary and its `id`.
  - `place_order(summary_id)`: refuses an unknown ID. It also refuses for `session.customer_turns <= summary.customer_turn_at_creation` ("Read the summary to the customer and wait for a clear yes."). On success, removes the summary.

- [ ] **Step 1: Write failing tests.** Cover each rule above. Include: `place_order` right after `prepare_order` fails. After `session.customer_turns += 1` it succeeds and calls `FakeStore.place_order` once. Quantity 0 and 1000 fail. Unknown handle fails with "No product has the handle". A guest `get_order_status` without email or phone fails and the fake store gets no call. Bad JSON arguments give `ok=False`.
- [ ] **Step 2: Run `uv run pytest tests/unit/test_tools.py`.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add session and store tools with order confirmation gate"`

---

### Task 6: Instructions

**Files:**
- Create: `src/support_agent/instructions.py`
- Test: `tests/unit/test_instructions.py`

**Interfaces:**
- Produces `build_instructions(channel: Literal["chat","voice"], identity: Identity, now: datetime) -> str`.
- Content rules: store name "Robo Parts BD", Bangladesh, BDT, cash on delivery only. Answer in the customer's language. Product names and spec values in English. Facts only from tools. Never place an order without the `prepare_order` summary read back and a clear yes. Delivery options and fees come from `prepare_order`. Unsupported requests sentence from Global Constraints. For a logged-in customer, the first name and that the account orders are available. The channel style rules.

- [ ] **Step 1: Write failing tests.** Assert: `voice` text contains "one to three short sentences" and "no lists". `chat` text does not. A customer identity puts "Rahim" in the text. A guest identity contains "not logged in". Both contain "cash on delivery" and "Bangla".
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add agent instructions with chat and voice styles"`

---

### Task 7: LLM client and agent loop

**Files:**
- Create: `src/support_agent/llm.py`, `src/support_agent/loop.py`
- Test: `tests/unit/test_loop.py` (scripted `FakeLlm`), `tests/llm/test_ollama_smoke.py` (marker `llm`)

**Interfaces:**
- Produces `LlmChunk(text: str | None, tool_calls: list[ToolCallDelta] | None, usage: dict | None)` and the protocol `LlmClient.stream(model: str, messages: list[dict], tools: list[dict]) -> AsyncIterator[LlmChunk]`. `OpenAiCompatClient(settings)` implements it with the `openai` SDK, `stream=True`, and assembles tool call deltas by index. `list_models() -> list[str]` reads `GET {base}/models`.
- Produces `AgentEvent(type: Literal["turn_start","llm_start","first_token","text","llm_end","tool_start","tool_end","answer","error"], t_ms: float, data: dict)`, where `t_ms` is the time since `turn_start`.
- Produces `run_turn(session: Session, user_text: str, llm: LlmClient, runner: ToolRunner, model: str, max_tool_rounds: int) -> AsyncIterator[AgentEvent]`. It increments `session.customer_turns`, appends the user message, and loops: stream the model. On tool calls, run each tool, append results, repeat. Stop at a text answer or at `max_tool_rounds` (then emit `error` "The agent used too many tool calls."). `llm_end` data holds `ms`, `first_token_ms`, `tokens` (from the usage data), and `tokens_per_s`. Any exception from the model client gives one `error` event and ends the turn.

- [ ] **Step 1: Write failing unit tests with `FakeLlm`.** Cases: plain answer gives events in order `turn_start, llm_start, first_token, text…, llm_end, answer`. One tool call gives `tool_start`, `tool_end`, then a second `llm_start`. Seven rounds of tool calls with `max_tool_rounds=6` end with `error`. A `FakeLlm` that raises `httpx.ConnectError` gives one `error` event with "model server". `session.customer_turns` grows by 1 per turn.
- [ ] **Step 2: Run `uv run pytest tests/unit/test_loop.py`.** Expected: FAIL.
- [ ] **Step 3: Implement both files.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Write the live smoke test (marker `llm`).** The test uses `qwen3.5:4b` and the live store. The message "Do you have an ESP32 board?" gives at least one `search_products` call and an answer that contains "ESP32".
- [ ] **Step 6: Run `uv run pytest -m llm`.** Expected: PASS. Record the `first_token_ms` values in the commit message.
- [ ] **Step 7: Commit.** `git commit -m "feat: add streaming agent loop with timed events"`

---

### Task 8: Chat server

**Files:**
- Create: `src/support_agent/server/app.py`, `src/support_agent/__main__.py` (runs uvicorn on port 8100)
- Test: `tests/unit/test_server.py` (FastAPI `TestClient`, `FakeLlm`, `FakeStore`)

**Interfaces:**
- Produces routes:
  - `POST /api/session` body `{channel}` returns `{session_id}` as a guest.
  - `POST /api/login` body `{session_id, email, password}` returns `{first_name}`. 401 with `{"message": "The email or the password is wrong."}` on failure.
  - `POST /api/logout` body `{session_id}`.
  - `POST /api/chat` body `{session_id, message, model?, channel?}` returns `text/event-stream`, one `data:` JSON line per `AgentEvent`.
  - `GET /api/models` returns `{models, default}`.
  - `GET /` serves `static/index.html`.
- Sessions live in memory in a dict. An unknown session ID gives 404.

- [ ] **Step 1: Write failing tests.** Session create. Login success and 401. Chat stream contains `turn_start` and `answer`. Unknown session 404. `channel` in a chat request changes the session channel.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: add chat server with streamed agent events"`

---

### Task 9: Chat page and debug panel

**Files:**
- Create: `src/support_agent/server/static/index.html` (HTML, CSS, and JavaScript in one file)

**Interfaces:**
- Consumes: Task 8 routes.
- Layout: left, the chat with a login form (email, password, "Log in", "Chat as guest") and a message box. Right, the debug panel with a model menu, a channel switch (Chat / Voice), a "New conversation" button, and a table of events per turn: step, model, tool name, input, output (collapsed), ms, first token ms, tokens per second, and the turn total.

- [ ] **Step 1: Build the page.**
- [ ] **Step 2: Start the server with `uv run python -m support_agent` and check in a browser.** Log in as `rahim@example.com`. Ask "What are my orders?". Make sure that the answer lists orders 1 and 6 and the panel shows a `list_my_orders` row with times. Chat as a guest and ask for order 6 with `rahim@example.com`. Switch to Voice and make sure that the next answer has no list.
- [ ] **Step 3: Commit.** `git commit -m "feat: add chat page with debug panel"`

---

### Task 10: Benchmark

**Files:**
- Create: `src/support_agent/bench/runner.py`, `src/support_agent/bench/cases/*.yaml` (at least 24 cases)
- Test: `tests/unit/test_bench.py` (case loading and checks with `FakeLlm`)

**Interfaces:**
- Case YAML: `id`, `language: bn | en | mixed`, `login: rahim | none`, `channel: chat | voice`, `turns: [ {user, expect: {tools_called?: [names], tools_not_called?: [names], answer_contains_any?: [strings], answer_not_contains?: [strings], order_placed?: bool}} ]`. A string `{live:price:<handle>}` in an expectation becomes the live price at run time.
- Command: `uv run python -m support_agent.bench --model <name> [--cases <glob>] [--out reports/]`. It writes `reports/<date>-<model>.md` and `.json`. The report has pass or fail per case and turn, `first_token_ms` and turn time (median and 90th percentile), and tokens per second. It also has the peak model memory from `GET http://localhost:11434/api/ps` (`size` in bytes) polled every second.
- Case coverage, with the number of cases:
  - Product search by need, in each language: 8.
  - Product detail and price: 3.
  - Out-of-stock product: 2.
  - Logged-in orders and status: 3.
  - Guest order lookup with and without proof: 3.
  - Tracking number: 1.
  - Order placement with a yes: 2.
  - Order summary answered with no or a change: 2 (Review Focus 1).
  - Unsupported request, for example cancel: 1.
  - Voice channel style: 1. The answer has no line that starts with "-" or "1.".
- Guest order emails: `bench-<run id>@example.com`.

- [ ] **Step 1: Write failing unit tests for the loader and the checks.**
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement the runner and write the cases.**
- [ ] **Step 4: Run the unit tests.** Expected: PASS.
- [ ] **Step 5: Run the benchmark with `qwen3.5:4b`.** Expected: a report file. Record the pass rate and the median first token time.
- [ ] **Step 6: Commit.** `git commit -m "feat: add benchmark with scripted conversations"`

---

### Task 11: Model comparison, default model, and README

**Files:**
- Create: `agent/README.md`, `agent/reports/` (committed reports)
- Modify: `agent/.env.example` (the chosen `LLM_MODEL`)

- [ ] **Step 1: Pull at least two more models in Ollama.** Each one must fit 5 to 9 GB and support tool calls. Run the benchmark for each, plus `qwen3.5:4b`. Run `qwen3.8:27b-mlx` once as the quality reference. If memory is short, stop the store containers for that run.
- [ ] **Step 2: Write `reports/2026-11-15-model-comparison.md`.** One table: model, size, peak memory, pass rate, pass rate per language, median and 90th percentile first token ms, median turn ms, tokens per second.
- [ ] **Step 3: Choose the default model.** Rule: the fastest model that passes at least 90 percent and stays under 12 GB. If no model passes 90 percent, choose the highest pass rate under 12 GB, and write the gap in the report.
- [ ] **Step 4: Write the README.** Sections: what it is, requirements (Ollama, uv, the demo store), and start. Then the chat page, the debug panel, and the benchmark. Last, the configuration, the test commands, and the model comparison result.
- [ ] **Step 5: Run all tests.** `uv run pytest` (unit, store, llm). Expected: all PASS.
- [ ] **Step 6: Commit.** `git commit -m "docs: add README and model comparison"`
