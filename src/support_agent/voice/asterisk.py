"""The Asterisk chan_websocket protocol: the MEDIA_START parser and the frame serializer.

Asterisk 22.11 chan_websocket facts (docs.asterisk.org, "WebSocket Channel Driver"):

- Asterisk is the WebSocket client. The first message is the TEXT message MEDIA_START.
- The default control format is plain text: "MEDIA_START key:value key:value ...".
  The JSON format (option f(json) or chan_websocket.conf) sends an object with an
  "event" key and the same fields, plus "channel_variables".
- Media is BINARY messages of raw audio. The codec slin16 is signed 16-bit
  little-endian mono PCM at 16000 Hz. One 20 ms frame is 640 bytes.
- The agent sends commands as TEXT messages. FLUSH_MEDIA drops the queued agent audio.
  HANGUP ends the call.

The server imports routes.py at start, and routes.py imports this module only at the
first phone call. This module imports Pipecat, so it must stay out of the server start.
"""

import json
import re
from dataclasses import dataclass

from loguru import logger
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

# The codec of the dial string WebSocket/agent/c(slin16).
SLIN16 = "slin16"
SLIN16_RATE = 16000
# 16-bit samples.
SAMPLE_BYTES = 2
# The JSON channel variables that can hold the caller number.
CALLER_VARIABLES = ("CALLERID(num)", "CALLERID_NUM", "caller")
# A caller number for the debug hub: digits with an optional "+". Other text becomes
# "phone", so that the hub never shows a name or an email.
CALLER_NUMBER = re.compile(r"^\+?[0-9]{1,20}$")


@dataclass(frozen=True)
class MediaStart:
    """The fields of MEDIA_START that the agent uses."""

    format: str
    optimal_frame_size: int
    ptime: int
    channel: str
    caller: str | None


def parse_control(text: str) -> tuple[str, dict]:
    """Returns the event name and the fields of a control message, plain text or JSON."""
    text = text.strip()
    if text.startswith("{"):
        try:
            message = json.loads(text)
        except ValueError:
            return "", {}
        if not isinstance(message, dict):
            return "", {}
        return str(message.get("event", "")), message
    name, _, rest = text.partition(" ")
    fields = {}
    for part in rest.split():
        key, sep, value = part.partition(":")
        if sep:
            fields[key] = value
    return name, fields


def parse_media_start(text: str) -> MediaStart | None:
    """Returns the MEDIA_START fields, or None if the message is not a valid MEDIA_START."""
    name, fields = parse_control(text)
    if name != "MEDIA_START":
        return None
    try:
        frame_size = int(fields.get("optimal_frame_size", 0))
        ptime = int(fields.get("ptime", 20))
    except (TypeError, ValueError):
        return None
    variables = fields.get("channel_variables")
    caller = None
    if isinstance(variables, dict):
        caller = next((str(variables[key]) for key in CALLER_VARIABLES if variables.get(key)),
                      None)
    return MediaStart(format=str(fields.get("format", "")), optimal_frame_size=frame_size,
                      ptime=ptime, channel=str(fields.get("channel", "")), caller=caller)


def caller_from(start: MediaStart, query_caller: str | None) -> str:
    """The caller for the debug hub: the caller number from Asterisk, or "phone".

    MEDIA_START gives the number only in the JSON format. The dialplan can also send it
    as the URI parameter "caller" with the dial option v(caller=...).
    """
    for value in (start.caller, query_caller):
        if value and CALLER_NUMBER.match(value):
            return value
    return "phone"


class AsteriskFrameSerializer(FrameSerializer):
    """Pipecat serializer for the Asterisk chan_websocket protocol with slin16 audio.

    The pipeline of a phone call runs at 16000 Hz in and out, the rate of slin16. So the
    serializer does not resample the caller audio. The output transport sends the agent
    audio in frames of optimal_frame_size bytes and paces them in real time.
    """

    def __init__(self, **kwargs):
        super().__init__(FrameSerializer.InputParams(audio_in_sample_rate=SLIN16_RATE,
                                                     audio_out_sample_rate=SLIN16_RATE),
                         **kwargs)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            # The caller started to speak. Asterisk must drop the agent audio in its queue.
            await self._reset_audio_out()
            return "FLUSH_MEDIA"
        if isinstance(frame, EndFrame):
            return "HANGUP"
        if isinstance(frame, AudioRawFrame):
            if frame.sample_rate == SLIN16_RATE:
                return frame.audio
            return await self._resample_audio_out(frame) or None
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            # A partial sample cannot be played. Asterisk sends whole frames.
            if not data or len(data) % SAMPLE_BYTES:
                return None
            return InputAudioRawFrame(audio=data, sample_rate=SLIN16_RATE, num_channels=1)
        name, _fields = parse_control(data)
        if name == "HANGUP":
            return EndFrame()
        if name == "MEDIA_XOFF":
            logger.warning("Asterisk sent MEDIA_XOFF: its queue of agent audio is full.")
        elif name == "MEDIA_XON":
            logger.info("Asterisk sent MEDIA_XON: its queue of agent audio has space again.")
        elif name:
            logger.debug(f"Asterisk control message: {name}")
        return None
