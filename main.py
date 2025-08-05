from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Atau [""] untuk semua origin (tidak direkomendasikan di production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(diarization_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
