from datetime import datetime

from support_agent.instructions import build_instructions
from support_agent.session import Identity
from tests.unit.fakes import RAHIM

NOW = datetime(2026, 11, 15, 10, 0)
CUSTOMER = Identity(kind="customer", token="t", customer=RAHIM)


def test_voice_style_rules():
    text = build_instructions("voice", Identity.guest(), NOW)
    assert "one to three short sentences" in text
    assert "no lists" in text


def test_chat_has_no_voice_rules():
    assert "one to three short sentences" not in build_instructions("chat", Identity.guest(), NOW)


def test_customer_identity():
    text = build_instructions("chat", CUSTOMER, NOW)
    assert "Rahim" in text
    assert "logged in" in text


def test_guest_identity():
    assert "not logged in" in build_instructions("chat", Identity.guest(), NOW)


def test_common_rules():
    for channel in ("chat", "voice"):
        text = build_instructions(channel, Identity.guest(), NOW)
        assert "cash on delivery" in text
        assert "Bangla" in text
        assert "prepare_order" in text
        assert "Robo Parts BD" in text


def test_summary_and_total_only_from_prepare_order():
    text = build_instructions("chat", Identity.guest(), NOW)
    assert "Never write an order summary or a total yourself" in text


def test_filters_only_for_stated_needs():
    text = build_instructions("chat", Identity.guest(), NOW)
    assert "Use a spec filter only for a need that the customer stated" in text


def test_voice_instructions_force_english():
    text = build_instructions("voice", Identity.guest(), NOW)
    assert "Answer in English" in text
    assert "language of the customer's last message" not in text
    assert "digit by digit" in text
    assert "phone number" in text  # guests give the phone number for order status


def test_chat_instructions_keep_language_rule():
    assert "language of the customer's last message" in build_instructions(
        "chat", Identity.guest(), NOW
    )
