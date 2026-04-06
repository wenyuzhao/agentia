from dataclasses import dataclass


@dataclass
class LiveChunkAudio:
    data: bytes
    mime_type: str = "audio/pcm;rate=16000"


@dataclass
class LiveChunkVideo:
    data: bytes
    mime_type: str = "image/jpeg"


@dataclass
class LiveChunkImage:
    data: bytes
    mime_type: str = "image/jpeg"


@dataclass
class LiveChunkText:
    text: str


@dataclass
class LiveChunkEnd: ...


LiveChunk = (
    LiveChunkAudio | LiveChunkVideo | LiveChunkImage | LiveChunkText | LiveChunkEnd
)
