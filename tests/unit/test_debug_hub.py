import asyncio

from support_agent.voice.debug_hub import DebugHub


async def test_hub_streams_events_to_subscribers():
    hub = DebugHub()
    stream = hub.subscribe()
    first = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    call_id = hub.start_call("phone", "1001")
    hub.publish(call_id, "pause", {"ms": 900})
    start = await asyncio.wait_for(first, 1)
    event = await asyncio.wait_for(anext(stream), 1)
    assert start["kind"] == "call_start"
    assert start["call_id"] == call_id
    assert start["data"] == {"channel": "phone", "caller": "1001"}
    assert event["kind"] == "pause"
    assert event["data"] == {"ms": 900}
    hub.end_call(call_id)
    end = await asyncio.wait_for(anext(stream), 1)
    assert end["kind"] == "call_end"
    await stream.aclose()


def test_hub_keeps_last_20_calls():
    hub = DebugHub()
    ids = [hub.start_call("browser", f"c{i}") for i in range(25)]
    calls = hub.calls()
    assert len(calls) == 20
    assert calls[-1]["call_id"] == ids[-1]
    assert calls[0]["call_id"] == ids[5]


def test_calls_hold_events_and_end_time():
    hub = DebugHub()
    call_id = hub.start_call("phone", "1001")
    hub.publish(call_id, "stt", {"ms": 5})
    hub.end_call(call_id)
    call = hub.calls()[0]
    assert call["channel"] == "phone"
    assert [e["kind"] for e in call["events"]] == ["call_start", "stt", "call_end"]
    assert call["ended_at"] is not None


def test_publish_to_unknown_call_is_ignored():
    DebugHub().publish("nope", "x", {})


async def test_slow_subscriber_drops_oldest_and_never_blocks():
    hub = DebugHub(queue_size=3)
    stream = hub.subscribe()
    first = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    call_id = hub.start_call("browser", "x")
    await first
    for i in range(10):
        hub.publish(call_id, "tick", {"i": i})
    got = [(await asyncio.wait_for(anext(stream), 1))["data"]["i"] for _ in range(3)]
    assert got == [7, 8, 9]
    await stream.aclose()


def test_tool_data_hides_the_local_part_of_emails():
    hub = DebugHub()
    call_id = hub.start_call("browser", "guest")
    hub.publish(call_id, "tool_start", {
        "name": "prepare_order",
        "arguments": '{"email": "rahim.k+1@example.com", "address": {"city": "Dhaka", '
                     '"phone": "01712345678"}}',
    })
    hub.publish(call_id, "tool_end", {
        "name": "prepare_order", "ok": True,
        "content": '{"email": "voice-8801712345678@guest.robopartsbd.local", "total": 630}',
    })
    start, end = hub.calls()[0]["events"][1:]
    assert start["data"]["arguments"] == (
        '{"email": "***@example.com", "address": {"city": "Dhaka", "phone": "01712345678"}}'
    )
    assert end["data"]["content"] == '{"email": "***@guest.robopartsbd.local", "total": 630}'
    assert end["data"]["name"] == "prepare_order" and end["data"]["ok"] is True
