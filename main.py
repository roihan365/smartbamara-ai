from fastapi import FastAPI, UploadFile, File
from diarization import (
    get_user,
    # process_transcription_with_diarization,
    upload_audio,
    transcribe_audio_api,
    get_result
)
import uvicorn

from diarization import router as diarization_router

app = FastAPI()
app.include_router(diarization_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
