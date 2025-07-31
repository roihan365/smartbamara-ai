from pydantic import BaseModel
from typing import List, Optional, Any

class DiarizationSegment(BaseModel):
    start: str
    end: str
    speaker: str
    text: str

class TranscriptionResult(BaseModel):
    uuid: str
    status: str
    transcription: Optional[str]
    summary: Optional[Any]
    diarization: Optional[List[DiarizationSegment]]
    created_at: Optional[str]
