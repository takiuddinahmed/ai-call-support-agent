# Voice call benchmark at 16000 Hz (the main path: browser and G.722 phone)

Date: 2026-11-15. Model: qwen3.5:4b-q4_K_M (reasoning low), prompt budget 3200 estimated tokens. Speech-to-text: mlx-community/whisper-large-v3-turbo. Run ID: 51472f25.
Agent voice: af_heart. Caller voice: am_adam. Run time: 714.0 s.
Ollama context: 4096 tokens. Setup order: 58. The bench placed the setup order through the store client for the status turns.

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
| Pause median, turns without tools | 1.5 s or less | 2.99 s | missed |
| Filler start median, turns with tools | 1.5 s or less | 1.98 s | missed |
| Memory upper figure | 12 GB or less | 9.1 GB | met |
| Calls passed | all | 3 of 5 | missed |

The memory upper figure is the agent footprint, plus the runner footprint, plus the
Ollama /api/ps estimate. The last two can count the same weights two times.

## Pause

The pause is the time to the first agent audio after the end of the caller audio. In
a turn with tools, the first agent audio is the filler. The answer start is the time
to the answer audio.

Split turns: 2. A split turn gave more than one agent turn, or the agent spoke before the caller audio ended. The rows without tools, with tools, and all turns include the split turns. The split turns also have their own row. Turns where the agent spoke over the caller: 2.

| Turns | Count | Pause median (s) | Pause p90 (s) | Answer median (s) | Answer p90 (s) |
|---|---|---|---|---|---|
| Without tools | 14 | 2.99 | 3.60 | 2.99 | 3.60 |
| With tools | 20 | 1.98 | 3.77 | 4.65 | 8.48 |
| Split turns | 2 | 3.45 | 5.49 | 6.24 | 8.71 |
| All turns | 34 | 2.43 | 3.60 | 3.82 | 7.04 |

| Call | Turn | Tools | Pause (s) | Answer (s) | Timer pause (ms) | Split | Heard |
|---|---|---|---|---|---|---|---|
| voice-product-question | 1 | - | 3.60 | 3.60 | 3420 | no | Hello, can you help me find a part? |
| voice-product-question | 2 | search_products | 2.09 | 8.38 | 1947 | no | How much is the ESB32 dev kit? |
| voice-product-question | 3 | - | 1.82 | 1.82 | 1653 | no | Okay, thank you, that is all for now. |
| voice-guest-order-yes | 1 | search_products | 3.17 | 6.05 | 3004 | no | I want to order one breadboard 830. |
| voice-guest-order-yes | 2 | get_product, prepare_order | 3.60 | 7.37 | 3416 | no | My name is Karim Ali. |
| voice-guest-order-yes | 3 | - | 2.36 | 2.36 | 2199 | no | The address is House 5 Road 2 in Sillet. |
| voice-guest-order-yes | 4 | prepare_order | 5.33 | 10.40 | 5210 | no | My phone number is 018-1111-1111. |
| voice-guest-order-yes | 5 | place_order | 2.29 | 3.75 | 2110 | no | Yes, please place the order. |
| voice-guest-order-status | 1 | - | 3.47 | 3.47 | 3246 | no | Hi, I want to check the status of my order. |
| voice-guest-order-status | 2 | - | 3.59 | 3.59 | 3408 | no | The order number is 58. |
| voice-guest-order-status | 3 | get_order_status | 2.30 | 6.28 | 2176 | no | My phone number is 018-1111-1111. |
| voice-natural-long-turns | 1 | - | 3.40 | 3.40 | 3222 | no | Hi there, I ordered a breadboard from your shop a little while ago. and I want to know if it is shipped yet. |
| voice-natural-long-turns | 2 | get_order_status, get_order_status | 0.90 | 3.16 | 2325 | yes | Sure, the order number is 58. and the phone number on the order is 01811. 1 1 1 1 1 1 1 |
| voice-natural-long-turns | 3 | search_products | 6.00 | 9.32 | 1760 | yes | Okay, thanks. While I have you, I also need a distance sensor for my robot. How much is the HCSR04 ultrasonic sensor? |
| voice-long-call-memory | 1 | search_products | 2.34 | 4.42 | 2135 | no | Hello, I need some parts for a robot. |
| voice-long-call-memory | 2 | search_products | 1.83 | 4.67 | 1689 | no | Do you have the Arduino Uno? |
| voice-long-call-memory | 3 | - | 2.74 | 2.74 | 2584 | no | How much is the Arduino Uno? |
| voice-long-call-memory | 4 | search_products | 1.86 | 4.12 | 1700 | no | Is the Raspberry Pi 5 in stock? |
| voice-long-call-memory | 5 | - | 5.00 | 5.00 | 4822 | no | What memory sizes does it have? |
| voice-long-call-memory | 6 | search_products | 1.81 | 3.11 | 1632 | no | Do you sell servo motors? |
| voice-long-call-memory | 7 | - | 3.04 | 3.04 | 2876 | no | How much is the SG90 servo? |
| voice-long-call-memory | 8 | search_products | 1.76 | 4.16 | 1601 | no | Do you have an ultrasonic sensor? |
| voice-long-call-memory | 9 | search_products | 1.67 | 3.37 | 1488 | no | Do you have a motor driver? |
| voice-long-call-memory | 10 | - | 3.51 | 3.51 | 3334 | no | How much is the L298N motor driver? |
| voice-long-call-memory | 11 | search_products | 1.63 | 4.53 | 1447 | no | Do you have a breadboard? |
| voice-long-call-memory | 12 | - | 2.81 | 2.81 | 2645 | no | What is the price of the breadboard 830? |
| voice-long-call-memory | 13 | search_products | 1.69 | 3.90 | 1511 | no | Do you have jumper wires? |
| voice-long-call-memory | 14 | search_products | 1.78 | 4.62 | 1596 | no | How much is the DHT11 sensor? |
| voice-long-call-memory | 15 | - | 2.76 | 2.76 | 2615 | no | I also want to check an order. |
| voice-long-call-memory | 16 | get_order_status | 3.34 | 5.26 | 3159 | no | The order number is 58. |
| voice-long-call-memory | 17 | get_order_status | 1.87 | 4.75 | 1744 | no | My phone number is 018-1111-1111. |
| voice-long-call-memory | 18 | track_shipment | 2.49 | 5.68 | 2328 | no | Is it shipped yet? |
| voice-long-call-memory | 19 | - | 2.94 | 2.94 | 2775 | no | Do you deliver outside DACA? |
| voice-long-call-memory | 20 | - | 1.38 | 1.38 | 1194 | no | Okay, thank you, that is all. |

The timer pause is the TurnTimer value: the VAD stop to the first TTS audio frame.

## Filler

Turns with a filler: 20. Filler start median: 1.98 s. Filler start p90: 3.77 s.

| Call | Turn | Filler start (s) | Answer start (s) | Gap (s) |
|---|---|---|---|---|
| voice-product-question | 2 | 2.09 | 8.38 | 6.29 |
| voice-guest-order-yes | 1 | 3.17 | 6.05 | 2.88 |
| voice-guest-order-yes | 2 | 3.60 | 7.37 | 3.77 |
| voice-guest-order-yes | 4 | 5.33 | 10.40 | 5.06 |
| voice-guest-order-yes | 5 | 2.29 | 3.75 | 1.46 |
| voice-guest-order-status | 3 | 2.30 | 6.28 | 3.98 |
| voice-natural-long-turns | 2 | 0.90 | 3.16 | 2.25 |
| voice-natural-long-turns | 3 | 6.00 | 9.32 | 3.32 |
| voice-long-call-memory | 1 | 2.34 | 4.42 | 2.08 |
| voice-long-call-memory | 2 | 1.83 | 4.67 | 2.84 |
| voice-long-call-memory | 4 | 1.86 | 4.12 | 2.26 |
| voice-long-call-memory | 6 | 1.81 | 3.11 | 1.30 |
| voice-long-call-memory | 8 | 1.76 | 4.16 | 2.40 |
| voice-long-call-memory | 9 | 1.67 | 3.37 | 1.70 |
| voice-long-call-memory | 11 | 1.63 | 4.53 | 2.90 |
| voice-long-call-memory | 13 | 1.69 | 3.90 | 2.21 |
| voice-long-call-memory | 14 | 1.78 | 4.62 | 2.85 |
| voice-long-call-memory | 16 | 3.34 | 5.26 | 1.92 |
| voice-long-call-memory | 17 | 1.87 | 4.75 | 2.88 |
| voice-long-call-memory | 18 | 2.49 | 5.68 | 3.19 |

## Calls

Passed: 3 of 5. The checks are the chat benchmark checks.
The benchmark changes number words in the answer to digits before the checks.
A number in a check does not match a longer number: "270" does not match "2700".

| Call | Result | Failures |
|---|---|---|
| voice-product-question | pass | - |
| voice-guest-order-yes | FAIL | turn 4: order summary city: Sillet, the caller said: Sylhet |
| voice-guest-order-status | pass | - |
| voice-natural-long-turns | FAIL | turn 2: answer has none of: ['pending', 'processing', 'placed', 'received', 'not yet', 'confirmed', 'not shipped'] |
| voice-long-call-memory | pass | - |

| Call | Turn | Said | Answer |
|---|---|---|---|
| voice-product-question | 1 | Hello can you help me find a part? | Hello! What specific part are you looking for? Please tell me the name or type of component you need, such as a sensor, motor, or board. |
| voice-product-question | 2 | How much is the ESP32 DevKit? | The ESP32 DevKit V1 costs five hundred fifty taka, and there are three other options available: the ESP32-S3 DevKitC-1 at one thousand one hundred fifty taka, and the ESP32-CAM with camera at eight hundred fifty taka. Which one would you like to know more about? |
| voice-product-question | 3 | Okay thank you that is all for now. | You're welcome! If you need help with anything else, feel free to ask. Have a great day! |
| voice-guest-order-yes | 1 | I want to order one breadboard 830. | I found the Breadboard 830 Points at one hundred twenty taka with two hundred sixty-seven units in stock. To place your order, please provide your name, address, city, and phone number. |
| voice-guest-order-yes | 2 | My name is Karim Ali. | I need your full address, city, and phone number to place your order. Please provide those details. |
| voice-guest-order-yes | 3 | The address is House 5 Road 2 in Sylhet. | I still need your phone number to place your order. Please provide it. |
| voice-guest-order-yes | 4 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | Your order summary is one Breadboard 830 Points at one hundred twenty taka, with outside Dhaka delivery fee of one hundred fifty taka, making the total two hundred seventy taka paid by cash on delivery to address House 5 Road 2 in Sillet. Do you confirm this order? |
| voice-guest-order-yes | 5 | Yes please place the order. | Your order number is fifty-nine. The total of two hundred seventy taka will be paid by cash on delivery. |
| voice-guest-order-status | 1 | Hi I want to check the status of my order. | To check your order status, please provide me with the order number and either the email or phone number associated with that order. |
| voice-guest-order-status | 2 | The order number is 58. | To check your order status, please provide me with either the email or phone number associated with order number 58. |
| voice-guest-order-status | 3 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | Your order number 58 is currently pending with a total of two hundred seventy taka for one Breadboard 830 Points. The package has been picked up and is being delivered outside Dhaka. |
| voice-natural-long-turns | 1 | Hi there, I ordered a breadboard from your shop a little while ago, and I want to know if it has shipped yet. | To check your order status, please provide me with the order number and the phone number associated with that order. |
| voice-natural-long-turns | 2 | Sure, the order number is 58, and the phone number on the order is 0 1 8 1 1, 1 1 1, 1 1 1 1. | To check your order status, please provide me with the email or phone number associated with order number 58. / I'm sorry, but I couldn't find any order with the number 58 associated with the phone number 01811. Could you please double-check the order number or provide the email address used for the order? / I'm sorry, I couldn't find an order with the number 58 associated with the phone number 01811. Could you please clarify what "1 1 1" means? Or could you provide the email address used for the order instead? / I'm sorry, but "1 1 1 1" doesn't appear to be a valid order number or phone number. Could you please check your order details again? If you'd like to track your breadboard order, I'll need the correct order number and the email address or phone number used for that order. |
| voice-natural-long-turns | 3 | Okay, thanks. While I have you, I also need a distance sensor for my robot. How much is the HC SR04 ultrasonic sensor? | You're welcome! If you need any help with your order or have other questions about our products, feel free to ask. / The HC-SR04 ultrasonic distance sensor costs one hundred twenty taka. It has two hundred ninety-eight units in stock. Would you like me to help you with anything else? |
| voice-long-call-memory | 1 | Hello I need some parts for a robot. | I found several motor options for your robot project. Here are a few choices:  1. NEMA 17 Stepper Motor (42 mm, 1.7 A) - one thousand two hundred fifty taka 2. MG996R Metal Gear Servo Motor - four hundred eighty taka 3. L298N Dual H-Bridge Motor Driver - two hundred twenty taka  What other parts do you need for your robot? Do you require sensors, a microcontroller board, or additional components? |
| voice-long-call-memory | 2 | Do you have the Arduino Uno? | Yes, I have the Arduino Uno R3 (Compatible, ATmega328P) available at eight hundred fifty taka with eighty-eight units in stock. This microcontroller has twenty GPIO pins and supports various interfaces including I2C, SPI, UART, and USB.  Do you need any other components for your robot project? Perhaps sensors or additional motor drivers? |
| voice-long-call-memory | 3 | How much is the Arduino Uno? | The Arduino Uno R3 is eight hundred fifty taka. Would you like me to help you with any other parts for your robot project? |
| voice-long-call-memory | 4 | Is the Raspberry Pi 5 in stock? | Yes, the Raspberry Pi 5 is in stock. We have three options available: two gigabyte version for eight thousand five hundred taka with twelve units, four gigabyte version for ten thousand five hundred taka with nineteen units, and eight gigabyte version for thirteen thousand five hundred taka with eight units. Would you like me to help you with anything else for your project? |
| voice-long-call-memory | 5 | What memory sizes does it have? | The Raspberry Pi 5 comes in three memory sizes: two gigabyte for eight thousand five hundred taka, four gigabyte for ten thousand five hundred taka, and eight gigabyte for thirteen thousand five hundred taka. All are currently in stock. Would you like to see any other parts for your project? |
| voice-long-call-memory | 6 | Do you sell servo motors? | Yes, we do sell servo motors. We have the MG996R Metal Gear Servo Motor at four hundred eighty taka and the SG90 Micro Servo Motor at one hundred sixty taka. Both are currently in stock. Would you like to know more about these or see other parts for your project? |
| voice-long-call-memory | 7 | How much is the SG90 servo? | The SG90 Micro Servo Motor costs one hundred sixty taka. Would you like to see more details about it or other parts for your project? |
| voice-long-call-memory | 8 | Do you have an ultrasonic sensor? | Yes, we have the HC-SR04 Ultrasonic Distance Sensor at one hundred twenty taka. It measures distance from two centimeters to four hundred centimeters with a five volt operating voltage. Would you like to see more details about it or other parts for your project? |
| voice-long-call-memory | 9 | Do you have a motor driver? | Yes, we have four motor drivers in stock. Here are two options: the A4988 Stepper Motor Driver for one hundred eighty taka with three point three volts logic level, or the L298N Dual H-Bridge Motor Driver for two hundred twenty taka with five volts logic level. Which one would you like to see more details about? |
| voice-long-call-memory | 10 | How much is the L298N motor driver? | The L298N Dual H-Bridge Motor Driver is two hundred twenty taka. Would you like to see more details about it or check another motor driver? |
| voice-long-call-memory | 11 | Do you have a breadboard? | The Breadboard 830 Points is one hundred twenty taka and we have two hundred sixty-six of them in stock. Would you like to see more details about it or check another item? |
| voice-long-call-memory | 12 | What is the price of the breadboard 830? | The breadboard 830 points is one hundred twenty taka. Would you like to see more details about it or check another item? |
| voice-long-call-memory | 13 | Do you have jumper wires? | The Jumper Wire Set with 120 pieces is two hundred twenty taka and we have two hundred forty-five of them in stock. Would you like to see more details about it or check another item? |
| voice-long-call-memory | 14 | How much is the DHT11 sensor? | The DHT11 Temperature and Humidity Sensor Module is one hundred fifty taka and we have two hundred of them in stock. Would you like to see more details about it or check another item? |
| voice-long-call-memory | 15 | I also want to check an order. | To check your order, please give me the order number and the phone number of the order. |
| voice-long-call-memory | 16 | The order number is 58. | To check your order, please give me the order number and the email or phone number of the order. |
| voice-long-call-memory | 17 | My phone number is 0 1 8 1 1 1 1 1 1 1 1. | Order fifty-eight is pending with a total of two hundred seventy taka for one Breadboard 830 Points. It's shipped outside Dhaka and currently in transit with tracking number DEMO-5606794825. The order was created today at five thirty-five PM. |
| voice-long-call-memory | 18 | Is it shipped yet? | Yes, your order has been shipped and is currently in transit. It was picked up today at five thirty-eight PM and is on its way to you. |
| voice-long-call-memory | 19 | Do you deliver outside Dhaka? | Yes, we deliver outside Dhaka to all cities in Bangladesh. |
| voice-long-call-memory | 20 | Okay thank you that is all. | You're welcome! If you have any more questions about our products or orders in the future, feel free to reach out. Have a great day! |

## Memory

Memory upper figure: 9.1 GB. The goal is 12 GB or less.

| Process | Peak of the run (GB) |
|---|---|
| Agent process, RSS | 1.53 |
| Agent process, macOS footprint | 4.38 |
| Ollama processes, RSS sum | 3.92 |
| Ollama runner, macOS footprint | 1.59 |
| Ollama /api/ps estimate | 3.15 |

Peaks during each call (GB). The upper figure adds the agent footprint, the runner
footprint, and the estimate.

| Call | Turns | Agent footprint | Runner footprint | Ollama RSS | Estimate | Upper figure |
|---|---|---|---|---|---|---|
| voice-product-question | 3 | 3.94 | 1.38 | 3.87 | 3.15 | 8.5 |
| voice-guest-order-yes | 5 | 4.07 | 1.43 | 3.92 | 3.15 | 8.7 |
| voice-guest-order-status | 3 | 4.13 | 1.43 | 3.92 | 3.15 | 8.7 |
| voice-natural-long-turns | 3 | 4.3 | 1.54 | 3.91 | 3.15 | 9.0 |
| voice-long-call-memory | 20 | 4.38 | 1.59 | 3.91 | 3.15 | 9.1 |

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
In this run: 2 split turns, and 2 turns where the agent spoke over the caller.

The order city: the store client cannot read the address of a placed order back,
because OrderView has no address. So the city check reads the address in the last
prepare_order result. place_order sends that address to the store.

Empty answers: 0. The count includes the turns where the agent said the fallback line "Sorry, I did not get that. Can you say it again?" in place of an empty model answer. The Ollama context is 4096 tokens. The agent keeps the prompt under 3200 estimated tokens.

Wrong spoken or stored facts in this run:

- voice-guest-order-yes: turn 4: order summary city: Sillet, the caller said: Sylhet

## Note

The caller audio is synthetic speech with no noise. A real caller gives more
speech-to-text errors. The 8 kHz run is a G.711 fallback test only. The main
path is 16 kHz.
