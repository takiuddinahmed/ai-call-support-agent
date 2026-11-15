# Voice call benchmark at 16000 Hz (the main path: browser and G.722 phone)

Date: 2026-11-15. Model: qwen3.5:4b-q4_K_M (reasoning low), reasoning after tools none, prompt budget 3200 estimated tokens. Speech-to-text: mlx-community/whisper-small-mlx. Run ID: 2e71d17f.
Agent voice: af_heart. Caller voice: am_adam. Run time: 444.2 s.
Ollama context: 4096 tokens. Setup order: 57. The bench placed the setup order through the store client for the status turns.

Sample: 5 calls, 34 turns. This is a single run, rough baseline. Run it more than one time before you compare two changes.

The benchmark plays the caller turns of calls.yaml as Kokoro audio through the real
pipeline (VAD, smart turn, Whisper, agent, Kokoro). FileAudioTransport sends the caller
audio in real time and waits until the answer audio of each turn ends.
All times are from the end of the caller audio, so they include the VAD stop time and
the smart-turn decision. The user chat server on port 8100 ran during the run. A user
test call at the same time can make a turn slow.

## Goals

Each goal is a goal, not a gate. A missed goal does not fail the run.

| Goal | Target | Result | Verdict |
|---|---|---|---|
| Pause median, turns without tools | 1.5 s or less | 2.73 s | missed |
| Filler start median, turns with tools | 1.5 s or less | 1.58 s | missed |
| Memory upper figure | 12 GB or less | 8.5 GB | met |
| Calls passed | all | 2 of 5 | missed |

The memory upper figure is the agent footprint, plus the runner footprint, plus the
Ollama /api/ps estimate. The last two can count the same weights two times.

## Pause

The pause is the time to the first agent audio after the end of the caller audio. In
a turn with tools, the first agent audio is the filler. The answer start is the time
to the answer audio.

Split turns: 5. A split turn gave more than one agent turn, or the agent spoke before the caller audio ended. The rows without tools, with tools, and all turns include the split turns. The split turns also have their own row. Turns where the agent spoke over the caller: 4.

| Turns | Count | Pause median (s) | Pause p90 (s) | Answer median (s) | Answer p90 (s) |
|---|---|---|---|---|---|
| Without tools | 20 | 2.73 | 4.03 | 2.73 | 4.03 |
| With tools | 14 | 1.58 | 1.96 | 3.11 | 9.63 |
| Split turns | 5 | 1.18 | 3.99 | 4.80 | 13.53 |
| All turns | 34 | 2.00 | 3.73 | 2.75 | 4.78 |

| Call | Turn | Tools | Pause (s) | Answer (s) | Timer pause (ms) | Split | Heard |
|---|---|---|---|---|---|---|---|
| voice-product-question | 1 | - | 1.80 | 1.80 | 1620 | no | Hello, can you help me find a part? |
| voice-product-question | 2 | search_products | 1.75 | - | 1586 | no | How much is the ESP32 Dev Kit? |
| voice-product-question | 3 | - | 2.76 | 2.76 | 2598 | yes | Okay, thank you, that is all for now. |
| voice-guest-order-yes | 1 | search_products, get_product, prepare_order | 1.93 | 4.60 | 1767 | no | I want to order one breadboard 830. |
| voice-guest-order-yes | 2 | - | 1.55 | 1.55 | 1393 | no | My name is Karim Ali. |
| voice-guest-order-yes | 3 | - | 2.41 | 2.41 | 2245 | no | The address is House 5 Road 2 in Silat. |
| voice-guest-order-yes | 4 | - | 3.95 | 3.95 | 3845 | no | My phone number is 0181111111. |
| voice-guest-order-yes | 5 | - | 7.44 | 7.44 | 7236 | no | Yes please, place the order. |
| voice-guest-order-status | 1 | - | 3.83 | 3.83 | 3630 | no | Hi, I want to check the status of my order. |
| voice-guest-order-status | 2 | - | 2.59 | 2.59 | 2406 | no | The order number is 57. |
| voice-guest-order-status | 3 | get_order_status | 2.19 | 3.31 | 2061 | no | My phone number is 0181111111. |
| voice-natural-long-turns | 1 | - | 1.18 | 1.18 | 992 | yes | Hi there, I ordered a breadboard from your shop a little while ago. and I want to know if it has shipped yet. |
| voice-natural-long-turns | 2 | get_order_status, get_order_status, get_order_status | 0.58 | 10.19 | 615 | yes | Sure, the order number is 57, and the phone number on the order is 01811. 111 1111. |
| voice-natural-long-turns | 3 | search_products | 0.80 | 15.76 | 2026 | yes | Okay, thanks. While I have you, I also need a distance sensor for my robot. How much is the HTSR-04 ultrasonic sensor? |
| voice-long-call-memory | 1 | - | 2.01 | 2.01 | 1809 | no | Hello I need some parts for a robot. |
| voice-long-call-memory | 2 | search_products | 1.78 | 2.90 | 1641 | no | Do you have the Arduino Uno? |
| voice-long-call-memory | 3 | - | 2.72 | 2.72 | 2560 | no | How much is the Arduino Uno? |
| voice-long-call-memory | 4 | search_products | 1.40 | - | 1220 | no | Is the Raspberry Pi 5 in stock? |
| voice-long-call-memory | 5 | - | 4.80 | 4.80 | 4597 | yes | What memory sizes does it have? |
| voice-long-call-memory | 6 | search_products | 1.43 | 2.59 | 1208 | no | Do you sell servo motors? |
| voice-long-call-memory | 7 | - | 2.36 | 2.36 | 2179 | no | How much is the SG90 servo? |
| voice-long-call-memory | 8 | search_products | 1.91 | 3.48 | 1746 | no | Do you have an ultrasonic sensor? |
| voice-long-call-memory | 9 | search_products | 1.60 | 2.73 | 1442 | no | Do you have a motor driver? |
| voice-long-call-memory | 10 | - | 3.24 | 3.24 | 3007 | no | How much is the L298N motor driver? |
| voice-long-call-memory | 11 | search_products | 1.55 | 2.79 | 1348 | no | Do you have a breadboard? |
| voice-long-call-memory | 12 | - | 2.59 | 2.59 | 2409 | no | What is the price of the breadboard 830? |
| voice-long-call-memory | 13 | search_products | 1.47 | 2.63 | 1281 | no | Do you have jumper wires? |
| voice-long-call-memory | 14 | search_products | 1.46 | 2.70 | 1274 | no | How much is the DHT11 sensor? |
| voice-long-call-memory | 15 | - | 2.74 | 2.74 | 2576 | no | I also want to check an order. |
| voice-long-call-memory | 16 | - | 3.49 | 3.49 | 3302 | no | The order number is 57. |
| voice-long-call-memory | 17 | get_order_status | 1.98 | 3.65 | 1876 | no | My phone number is 0181111111. |
| voice-long-call-memory | 18 | - | 3.46 | 3.46 | 3273 | no | Is it shipped yet? |
| voice-long-call-memory | 19 | - | 2.74 | 2.74 | 2576 | no | Do you deliver outside DACA? |
| voice-long-call-memory | 20 | - | 1.12 | 1.12 | 942 | no | Okay, thank you that is all. |

The timer pause is the TurnTimer value: the VAD stop to the first TTS audio frame.

## Filler

Turns with a filler: 14. Filler start median: 1.58 s. Filler start p90: 1.96 s.

| Call | Turn | Filler start (s) | Answer start (s) | Gap (s) |
|---|---|---|---|---|
| voice-product-question | 2 | 1.75 | - | - |
| voice-guest-order-yes | 1 | 1.93 | 4.60 | 2.67 |
| voice-guest-order-status | 3 | 2.19 | 3.31 | 1.12 |
| voice-natural-long-turns | 2 | 0.58 | 10.19 | 9.61 |
| voice-natural-long-turns | 3 | 0.80 | 15.76 | 14.96 |
| voice-long-call-memory | 2 | 1.78 | 2.90 | 1.12 |
| voice-long-call-memory | 4 | 1.40 | - | - |
| voice-long-call-memory | 6 | 1.43 | 2.59 | 1.16 |
| voice-long-call-memory | 8 | 1.91 | 3.48 | 1.58 |
| voice-long-call-memory | 9 | 1.60 | 2.73 | 1.12 |
| voice-long-call-memory | 11 | 1.55 | 2.79 | 1.24 |
| voice-long-call-memory | 13 | 1.47 | 2.63 | 1.16 |
| voice-long-call-memory | 14 | 1.46 | 2.70 | 1.24 |
| voice-long-call-memory | 17 | 1.98 | 3.65 | 1.67 |

## Calls

Passed: 2 of 5. The checks are the chat benchmark checks.
The benchmark changes number words in the answer to digits before the checks.
A number in a check does not match a longer number: "270" does not match "2700".

| Call | Result | Failures |
|---|---|---|
| voice-product-question | pass | - |
| voice-guest-order-yes | FAIL | turn 4: tool not called: prepare_order; turn 4: answer has none of: ['270']; turn 4: answer does not have: 150; turn 4: no order summary to check the city; turn 5: order placed: False, expected: True |
| voice-guest-order-status | FAIL | turn 3: answer has none of: ['pending', 'processing', 'placed', 'received', 'not yet', 'confirmed'] |
| voice-natural-long-turns | FAIL | turn 2: answer has none of: ['pending', 'processing', 'placed', 'received', 'not yet', 'confirmed', 'not shipped'] |
| voice-long-call-memory | pass | - |

| Call | Turn | Said | Answer |
|---|---|---|---|
| voice-product-question | 1 | Hello can you help me find a part? | Hello! What part are you looking for? Please tell me the name of the component you need. |
| voice-product-question | 2 | How much is the ESP32 DevKit? | The ESP32 DevKit V1 is five hundred fifty taka and the ESP32-S3 DevKitC-1 is one thousand one hundred fifty taka. Which one would you like to buy? |
| voice-product-question | 3 | Okay thank you that is all for now. | You're welcome! The ESP32 Dev Kit V1 is five hundred fifty taka and has plenty of stock. If you need anything else, feel free to ask. |
| voice-guest-order-yes | 1 | I want to order one breadboard 830. | I found the Breadboard 830 Points at one hundred twenty taka. Please give me your name, address, city, and phone number to proceed with the order. |
| voice-guest-order-yes | 2 | My name is Karim Ali. | Thank you Karim. Please give me your address, city, and phone number to proceed with the order. |
| voice-guest-order-yes | 3 | The address is House 5 Road 2 in Sylhet. | Thank you Karim. Please give me your city and phone number to complete the order. |
| voice-guest-order-yes | 4 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | Thank you Karim. Please give me your city to complete the order. |
| voice-guest-order-yes | 5 | Yes please place the order. | Thank you Karim. Please give me your city to complete the order. |
| voice-guest-order-status | 1 | Hi I want to check the status of my order. | To check your order status, please provide me with the order number and either the email address or mobile phone number associated with that order. |
| voice-guest-order-status | 2 | The order number is 57. | To check your order status, please provide me with either the email address or mobile phone number associated with order number 57. |
| voice-guest-order-status | 3 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | I couldn't find an order with the number 57 using that phone number. Please double-check the order number or provide the email address associated with it. |
| voice-natural-long-turns | 1 | Hi there, I ordered a breadboard from your shop a little while ago, and I want to know if it has shipped yet. | To check your order, please give me the order number and either the email or phone number used when you placed it. / To check your order status, please give me the order number and either the email or phone number used when you placed it. |
| voice-natural-long-turns | 2 | Sure, the order number is 57, and the phone number on the order is 0 1 8 1 1, 1 1 1, 1 1 1 1. | I couldn't find an order with that number and phone number. Could you please double-check the order number or provide the email address used for the order? / I still couldn't find an order with that number and phone number. Could you please provide the email address used for the order? / I still cannot find an order with that number and phone number. Could you please provide the email address used for the order? |
| voice-natural-long-turns | 3 | Okay, thanks. While I have you, I also need a distance sensor for my robot. How much is the HC SR04 ultrasonic sensor? | I'm sorry I couldn't find your order with that number and phone. If you need any help with products or placing a new order, just let me know. / I found two distance sensors: the VL53L0X Time-of-Flight sensor for four hundred fifty taka and the HC-SR04 Ultrasonic sensor for one hundred twenty taka. Which one would you like to add to your order? / The HC-SR04 Ultrasonic Distance Sensor is one hundred twenty taka. Did you mean that model, or are you looking for a different sensor? |
| voice-long-call-memory | 1 | Hello I need some parts for a robot. | Hello! I'd be happy to help you find parts for your robot. Could you tell me what type of robot you're building and what specific components you need? For example, do you need motors, sensors, or controllers? |
| voice-long-call-memory | 2 | Do you have the Arduino Uno? | Yes, we have the Arduino Uno R3 for eight hundred fifty taka. It has twenty GPIO pins and supports I2C, SPI, UART, and other interfaces. Do you need any other parts for your robot? |
| voice-long-call-memory | 3 | How much is the Arduino Uno? | The Arduino Uno R3 is eight hundred fifty taka and it's currently in stock. |
| voice-long-call-memory | 4 | Is the Raspberry Pi 5 in stock? | The Raspberry Pi 5 is available in three variants: two GB for ten thousand five hundred taka, four GB for thirteen thousand five hundred taka, and eight GB for fifteen thousand five hundred taka. All are currently in stock. Do you need the power supply as well? |
| voice-long-call-memory | 5 | What memory sizes does it have? | The Raspberry Pi 5 comes in three memory sizes: two gigabytes for eight thousand five hundred taka, four gigabytes for ten thousand five hundred taka, or eight gigabytes for thirteen thousand five hundred taka. |
| voice-long-call-memory | 6 | Do you sell servo motors? | Yes, we sell two types of servo motors. The MG996R Metal Gear Servo Motor costs four hundred eighty taka and the SG90 Micro Servo Motor is one hundred sixty taka. Which one are you interested in? |
| voice-long-call-memory | 7 | How much is the SG90 servo? | The SG90 servo motor is one hundred sixty taka. |
| voice-long-call-memory | 8 | Do you have an ultrasonic sensor? | Yes, we have the HC-SR04 ultrasonic sensor for one hundred twenty taka. |
| voice-long-call-memory | 9 | Do you have a motor driver? | Yes, we have four motor drivers in stock. Which one would you like to see more details about? |
| voice-long-call-memory | 10 | How much is the L298N motor driver? | The L298N motor driver costs two hundred twenty taka. Would you like me to show you more details about this product? |
| voice-long-call-memory | 11 | Do you have a breadboard? | The Breadboard 830 Points costs one hundred twenty taka. Would you like to see more details about this product? |
| voice-long-call-memory | 12 | What is the price of the breadboard 830? | The breadboard 830 points costs one hundred twenty taka. Would you like me to show you more details about this product? |
| voice-long-call-memory | 13 | Do you have jumper wires? | The jumper wire set costs two hundred twenty taka. Would you like me to show you more details about this product? |
| voice-long-call-memory | 14 | How much is the DHT11 sensor? | The DHT11 sensor costs one hundred fifty taka. Would you like to see more details about it? |
| voice-long-call-memory | 15 | I also want to check an order. | To check your order, please provide the order number and either the email or phone number associated with it. |
| voice-long-call-memory | 16 | The order number is 57. | To check your order, please provide the email or phone number associated with order number 57. |
| voice-long-call-memory | 17 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | I'm sorry, but there is no order found with the number 57 and the phone number 0181111111. Please double-check the details or try another order number. |
| voice-long-call-memory | 18 | Is it shipped yet? | I was unable to find an order with number 57 using your phone number 0181111111. Could you please double-check the order number or provide the email address associated with it? |
| voice-long-call-memory | 19 | Do you deliver outside Dhaka? | Yes, we deliver outside Dhaka to other cities in Bangladesh. |
| voice-long-call-memory | 20 | Okay thank you that is all. | You're welcome! Feel free to contact us anytime if you have any questions or need assistance with your orders. |

## Memory

Memory upper figure: 8.5 GB. The goal is 12 GB or less.

| Process | Peak of the run (GB) |
|---|---|
| Agent process, RSS | 1.28 |
| Agent process, macOS footprint | 3.3 |
| Ollama processes, RSS sum | 4.08 |
| Ollama runner, macOS footprint | 2.03 |
| Ollama /api/ps estimate | 3.15 |

Peaks during each call (GB). The upper figure adds the agent footprint, the runner
footprint, and the estimate.

| Call | Turns | Agent footprint | Runner footprint | Ollama RSS | Estimate | Upper figure |
|---|---|---|---|---|---|---|
| voice-product-question | 3 | 2.68 | 1.92 | 3.97 | 3.15 | 7.8 |
| voice-guest-order-yes | 5 | 2.81 | 1.97 | 4.08 | 3.15 | 7.9 |
| voice-guest-order-status | 3 | 2.71 | 1.97 | 3.92 | 3.15 | 7.8 |
| voice-natural-long-turns | 3 | 3.06 | 1.97 | 3.96 | 3.15 | 8.2 |
| voice-long-call-memory | 20 | 3.3 | 2.03 | 3.98 | 3.15 | 8.5 |

On macOS, the footprint of the llama.cpp runner can miss the mapped model weights,
as the model comparison report found. The RSS counts only the weight pages that are
in memory at that time. Use the Ollama numbers as a lower limit, not as the full
memory of the model. The /api/ps estimate is the Ollama estimate of the weights and
the context. The agent footprint includes the Metal memory of MLX Whisper. The agent
RSS can miss that memory. The Ollama RSS sum includes the Ollama app and server.

## Known findings

Split turns: the turn-end model (smart turn) ends a caller turn at a pause inside a
sentence. Then the agent answers the first part and speaks over the caller.
In run 69f1cb75 (16 kHz), the caller said "My name is Karim Ali. The address is House 5,
Road 2, Sylhet, postal code 3100. My phone number is 018 1111 1111." in one turn. The
turn-end model ended the turn after the first sentence. The agent answered while the
caller still spoke. The answer audio started 4.27 s before the end of the caller audio
(a pause of -4.27 s).
In this run: 5 split turns, and 4 turns where the agent spoke over the caller.

The order city: the store client cannot read the address of a placed order back,
because OrderView has no address. So the city check reads the address in the last
prepare_order result. place_order sends that address to the store.

Empty answers: 0. The count includes the turns where the agent said the fallback line "Sorry, I did not get that. Can you say it again?" in place of an empty model answer. The Ollama context is 4096 tokens. The agent keeps the prompt under 3200 estimated tokens.

Wrong spoken or stored facts in this run:

- voice-guest-order-yes: turn 4: answer does not have: 150

## Note

The caller audio is synthetic speech with no noise. A real caller gives more
speech-to-text errors. The 8 kHz run is a G.711 fallback test only. The main
path is 16 kHz.
