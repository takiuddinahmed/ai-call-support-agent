from support_agent.voice.bench.wer import normalize_for_wer, word_error_rate


def test_wer_exact_match():
    assert word_error_rate("price of the ESP32", "Price of the ESP32.") == 0.0


def test_wer_one_substitution_in_four_words():
    # 1 substitution and 1 insertion over 4 reference words
    assert word_error_rate("price of the ESP32", "price of the ESP 32") == 0.5


def test_wer_empty_heard():
    assert word_error_rate("yes please", "") == 1.0


def test_normalized_esp32_spacing():
    assert word_error_rate(normalize_for_wer("price of the ESP32"),
                           normalize_for_wer("price of the ESP 32")) == 0.0


def test_normalized_tracking_number():
    ref = normalize_for_wer("The tracking number is BD 778 4521.")
    assert word_error_rate(ref, normalize_for_wer("The tracking number is BD778-4521.")) == 0.0


def test_normalized_phone_number():
    ref = normalize_for_wer("Please send the message to 01811 556677.")
    heard = normalize_for_wer("Please send the message to 01811-556677.")
    assert word_error_rate(ref, heard) == 0.0


def test_normalized_real_wrong_word_counts():
    ref = normalize_for_wer("Send it to Agrabad, Chittagong.")
    assert word_error_rate(ref, normalize_for_wer("Send it to Agrabod, Chittagong.")) == 0.2


def test_normalized_wrong_digit_counts():
    ref = normalize_for_wer("Check order 5120.")
    assert word_error_rate(ref, normalize_for_wer("Check order 5121.")) > 0.0
