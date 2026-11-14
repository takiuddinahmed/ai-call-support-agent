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
