import pytest

from support_agent.config import Settings
from support_agent.llm import OpenAiCompatClient
from tests.unit.fakes import FakeLlm


async def sent_kwargs(settings, **stream_kwargs):
    """Returns the keyword arguments that the client sends to the model server."""
    client = OpenAiCompatClient(settings)
    sent = {}

    async def no_chunks():
        return
        yield

    async def create(**kwargs):
        sent.update(kwargs)
        return no_chunks()

    client._client.chat.completions.create = create
    _ = [c async for c in client.stream("m", [], [], **stream_kwargs)]
    return sent


@pytest.mark.parametrize(("configured", "argument", "expected"), [
    ("low", None, "low"),
    ("low", "none", "none"),
    ("", "high", "high"),
])
async def test_reasoning_effort_argument_overrides_the_configuration(configured, argument, expected):
    settings = Settings(_env_file=None, llm_reasoning_effort=configured)
    sent = await sent_kwargs(settings, reasoning_effort=argument)
    assert sent["reasoning_effort"] == expected


async def test_no_reasoning_effort_when_configuration_is_empty():
    sent = await sent_kwargs(Settings(_env_file=None, llm_reasoning_effort=""))
    assert "reasoning_effort" not in sent


async def test_fake_llm_records_reasoning_effort():
    llm = FakeLlm(["a", "b"])
    _ = [c async for c in llm.stream("m", [], [])]
    _ = [c async for c in llm.stream("m", [], [], reasoning_effort="none")]
    assert llm.efforts_seen == [None, "none"]
