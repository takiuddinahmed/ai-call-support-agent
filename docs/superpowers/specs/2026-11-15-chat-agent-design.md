# Chat support agent design

Date: 2026-11-15

## Intent

The main project is an AI customer service agent that answers voice calls for an online store. The demo store (Medusa, in the `demo-store/` folder) is the first store that the agent serves. It sells parts for embedded systems and robotics in Bangladesh.

This design covers the first phase: a text chat version of the agent. The chat agent and the later voice agent share the same core: the instructions, the store tools, and the store adapter. If the chat agent handles the customer needs correctly, the voice phase starts on a tested base.

The user has two reasons for the project:

1. To build a working support agent for calls in Bangla and English.
2. To learn how to run AI models locally on limited resources, and how to measure and improve their speed and memory use.

Because of the second reason, all models run locally on this Mac. The project uses no paid cloud AI service.

## The customer needs

A customer can do three things through the agent:

1. Find a product that matches a need, for example "a 5V distance sensor with I2C for my robot".
2. Place an order with cash on delivery.
3. Get the status of an order.

## Goals

- A chat page where a person can log in with a store account, or chat as a guest, and do the three things above.
- Answers in the language that the customer uses: Bangla, English, or a mix of both in one sentence.
- Correct facts. The agent reads every product, price, stock number, and order status from the store API. It never invents them.
- A debug panel next to the chat. It shows each model call, each tool call with its input and output, and the time of each step.
- A test set of scripted conversations, and a command that runs them and reports correctness and speed. The user runs it after each change to compare models and settings.

## Non-goals

- Speech, audio, and telephony. These come in the next phases.
- A chat button on the storefront pages.
- Cancellations, returns, refunds, and address changes after an order. The agent says that it cannot do these and tells the customer to contact the store team.
- Handover to a human agent.
- More than one conversation at a time. The design must not prevent it, but the project does not test or tune for it.
- Any change to the demo store code.

## Anti-goals

These results are failures, even when the chat technically works:

- The agent states a price, a stock number, a product, or an order status that the store API did not return.
- The agent places an order without a clear "yes" from the customer. The agent must first read back the items, the address, the delivery option, and the total.
- The agent shows order details to a person who did not prove that the order is theirs.
- The time of a step is not visible. The user cannot optimize what the panel does not show.
- The core depends on the chat page, so the voice phase must rewrite it.

## Constraints

- Hardware: Apple M5 Pro, 24 GB shared memory, 16-core GPU. The demo store containers use about 3 GB.
- Memory budget: all AI models together use at most 12 GB.
- Docker on macOS cannot use the Apple GPU. The models run directly on macOS. The agent server can run on macOS or in Docker and reach the models over the local network.
- Ollama 0.40.16 is installed. The models `qwen3.5:4b` (4 GB) and `qwen3.8:27b-mlx` (18 GB) are downloaded. The 27B model is over budget. It can serve only as a quality reference.
- The agent talks to the store only through the store HTTP API. It never reads the store database.
- No published test measures small local models on Bangla tool calls. The project must build its own test set.

## Approaches

Three ways to build the agent core were considered.

1. Own agent loop in Python (chosen). The loop sends the conversation and the tool list to the model, runs the tool calls, and repeats until the model answers. Every step is visible and timed. In the voice phase, Pipecat runs its own loop with the same tools, instructions, and store adapter.
2. An agent framework such as Pydantic AI or LangGraph. It gives retries, typed tools, and memory, but it hides the step timing, and Pipecat does not use it. The voice phase then needs a second integration.
3. One speech-to-speech model for the later voice phase. Today such models handle Bangla and tool calls poorly and need more memory. Rejected for the voice phase, and not relevant to chat.

The user chose approach 1.

For the later voice phase, the user agreed on Pipecat with local models. The user will reach the agent through an IP phone (a SIP phone or softphone on a local Asterisk server). That is out of scope here, but the core must support it.

## Decisions from the user

- The agent is a separate project from the demo store.
- Text chat comes first. Voice and the IP phone come later.
- Languages: Bangla and English.
- All AI runs locally on this Mac. Optimize for this Mac.
- One conversation at a time is enough.
- The chat identifies the customer through a store login on the chat page, not through a storefront chat button.
- The agent core is an own loop in Python, not a framework.

## Decisions from Claude (the user can change these)

### Project layout

- The workspace folder `ai-audio-call/` holds two separate git repositories:
  - `demo-store/`: the current store repository, moved into its own folder with its history.
  - `agent/`: a new repository for the agent.
- The store design and plan documents move with the store into `demo-store/docs/`.
- Docker Compose names a project after its folder. The move would make new, empty volumes, and the store would lose its data and its publishable key. The store `docker-compose.yml` therefore gets the line `name: ai-audio-call`, so that the containers and the volumes keep their names. This is a configuration change, not a store code change.

### Stack

- Python 3.12 or later, managed with `uv`.
- FastAPI for the agent server.
- The chat page is plain HTML and JavaScript, served by the agent server. Replies stream to the page as server-sent events (one-way messages from server to page).
- The agent calls the model through the OpenAI-compatible chat API. Ollama serves it at `http://localhost:11434/v1`. The base URL and the model name are settings in the configuration file, so that llama.cpp or an MLX server can replace Ollama for speed tests.
- The agent server runs directly on macOS for development. A Docker setup is optional and is left to the builder.

### Core parts

The core is a Python package with no web code in it. The chat server and the later voice app import it.

- Store adapter: the only code that calls the store API. It knows the publishable key, the region, and the URLs.
- Tools: the functions that the model can call. Each tool has a name, a description, an input schema, and a handler that uses the store adapter.
- Instructions: the system prompt. It has a channel setting:
  - `chat`: short answers, lists are allowed.
  - `voice`: one to three short sentences, no lists or tables, numbers and prices said in words.
  The chat page has a switch for the channel, so that the user can test the voice style in chat.
- Agent loop: runs one customer turn. It emits events for each step (model call start, first token, tool call, tool result, final answer), with times. The chat server sends these events to the debug panel.
- Session: holds the conversation history and the customer identity (logged-in customer, verified guest, or unknown guest).

### Tools

| Tool | Purpose |
| --- | --- |
| `search_products` | Find products by text, category, and specification values. |
| `get_product` | Get one product with its variants, prices, stock, and specifications. |
| `list_my_orders` | List the orders of the logged-in customer. |
| `get_order_status` | Get an order and its tracking. For a guest, it needs the order number and the email or phone. |
| `track_shipment` | Get the status history for a tracking number. |
| `prepare_order` | Check the order details and return a summary with the total and a summary ID. |
| `place_order` | Create the cart, add the items, the address, and the delivery option, and complete the order with cash on delivery. It needs a summary ID from `prepare_order`. |

Product search by specification: the store API cannot filter on specification values. The store adapter loads the product list with `metadata` once, keeps it in memory for five minutes, and filters it in the agent. The demo store has 55 products, so this is cheap. The tool accepts typed filters, for example `operating_voltage_max_v >= 5` and `interface contains I2C`, with the spec keys from the store README.

### Order placement rules

- The agent collects the items and quantities, the shipping address, the city, the phone, and, for a guest, the name and the email.
- For a logged-in customer, the agent offers the default address from the account.
- The delivery option follows the city: "Inside Dhaka" for Dhaka, "Outside Dhaka" for other cities. The store enforces the same rule.
- Before it calls `place_order`, the agent reads back the items, the quantities, the address, the delivery option, and the total.
- Then the agent asks for a clear "yes".
- The order step uses two tools, so that the code enforces the read-back:
  - `prepare_order` checks the items, the stock, the address, and the delivery option. It returns the summary with the total and a summary ID, and the session stores that summary.
  - `place_order` takes the summary ID. The code refuses to run it unless the session holds that exact summary and the customer sent at least one message after the summary.
- The model decides if that customer message means "yes". Word matching cannot cover Bangla, English, and mixed speech well. The test set includes "no" and "change the address" answers, to catch a model that orders too soon.
- For a logged-in customer, the order links to the customer account through the customer token.

### Identity and privacy

- A logged-in customer can see all of their own orders.
- A guest must give the order number and the email or phone. The agent uses the store order lookup route. A failed lookup gives the same answer as an unknown order.
- A tracking number alone is enough to get the shipment status and its history, as on the public store tracking page and at real couriers. It does not give the items, the address, or the customer details.
- For a logged-in customer, the agent gets the tracking of an order through the order lookup with the account email.
- The agent shows an order only after the session made sure that the order belongs to this person.

### Debug panel

- One row for each step of each turn: model call, time to first token, total model time, tokens per second, tool name, tool input, tool output, tool time.
- A model menu that lists the models from Ollama, so that the user can switch models between conversations.
- A total time for each turn, from the customer message to the end of the answer.

### Test set and benchmark

- A folder of scripted conversations in YAML (a plain text data format). Each one has the customer messages, the login state, and the expected results: the tool calls, and facts that the final answer must or must not contain.
- The set covers Bangla, English, and mixed messages, and the three customer needs, plus refusals such as a cancellation request.
- A command runs the whole set against a chosen model and writes a report: pass or fail for each conversation, the time of each step, the peak memory of the model server, and the totals.
- The first report compares at least three models in the memory budget, for example `qwen3.5:4b` and two models in the 5 to 9 GB range.
- The benchmark runs against the live demo store, and order tests place real orders. The guest orders use emails such as `bench-<run id>@example.com`, so that they are easy to find.
- Order tests use products with high stock. Expected facts, such as a price or a stock number, come from a store API read at run time, not from fixed values in the test files.

### Success numbers

- Correctness: at least 90 percent of the test conversations pass with the chosen model.
- Speed: the time to the first answer token is under 1 second for most turns without a tool call. This number prepares the 1.5-second voice target of the next phase.
- Memory: the chosen model stays under the 12 GB budget.

## Left to the builder

- The exact module layout inside the core package.
- The test framework and the test layout. The tests must cover the tools, the loop, and the server.
- The look of the chat page and the debug panel.
- The exact wording of the instructions. They must follow the rules above.
- The models in the first comparison. They must fit the budget.
- The YAML format of the test conversations.
