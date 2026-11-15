from support_agent.config import Settings
from support_agent.session import Identity, Session
from support_agent.tools import TOOL_SPECS
from support_agent.voice.deps import VoiceDeps
from tests.unit.fakes import FakeLlm, FakeStore


async def test_warm_up_sends_the_voice_prompt_and_the_tools():
    llm = FakeLlm(["Hello there, how can I help?"])
    deps = VoiceDeps(settings=Settings(_env_file=None), store=FakeStore(), llm=llm)
    session = Session.new(Identity.guest(), "voice")
    await deps.warm_up_llm(session)
    (messages,) = llm.seen
    assert messages[0]["role"] == "system" and "English" in messages[0]["content"]
    assert llm.tools_seen == [TOOL_SPECS]
    # The warm-up does not change the conversation.
    assert session.messages == [] and session.customer_turns == 0


async def test_warm_up_does_not_raise_when_the_model_fails():
    llm = FakeLlm([], error=RuntimeError("Ollama is not running"))
    deps = VoiceDeps(settings=Settings(_env_file=None), store=FakeStore(), llm=llm)
    await deps.warm_up_llm(Session.new(Identity.guest(), "voice"))
