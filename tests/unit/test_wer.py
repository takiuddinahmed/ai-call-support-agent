from support_agent.voice.bench.wer import word_error_rate


def test_wer_exact_match():
    assert word_error_rate("price of the ESP32", "Price of the ESP32.") == 0.0


def test_wer_one_substitution_in_four_words():
    # 1 substitution and 1 insertion over 4 reference words
    assert word_error_rate("price of the ESP32", "price of the ESP 32") == 0.5


def test_wer_empty_heard():
    assert word_error_rate("yes please", "") == 1.0
