from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import uuid
import subprocess
from regex import D
import torch
import torchaudio
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from huggingface_hub import login
from datetime import timedelta
import time
import numpy as np
from sklearn.cluster import SpectralClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from fastapi import Body
from pydub import AudioSegment
from playwright.sync_api import sync_playwright
import threading
from pydub.playback import play
from fastapi import APIRouter, UploadFile, File, Form
from gemini.summarize import ringkas_transkrip
from fastapi import BackgroundTasks
from result.result_store import init_result, save_result, get_result
from supabase_client import supabase

router = APIRouter()

# Load API Keys
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Token untuk Pyannote

if not HUGGINGFACE_TOKEN:
    raise ValueError("❌ HUGGINGFACE_TOKEN tidak ditemukan di environment variables!")

# Login ke Hugging Face untuk memastikan akses model
login(HUGGINGFACE_TOKEN)
print("Login berhasil!")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Model akan berjalan di: {DEVICE}")

app = FastAPI()

# 🔄 **Cache model agar tidak fetch setiap request**
diarization_pipeline = None
whisper_model = None
embedding_pipeline = None  # Model embedding Pyannote


def load_models():
    global diarization_pipeline, whisper_model, embedding_pipeline

    if diarization_pipeline is None:
        try:
            print("📥 Memuat model diarization...")
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", use_auth_token=HUGGINGFACE_TOKEN
            )

            embedding_pipeline = PretrainedSpeakerEmbedding(
                "pyannote/embedding", use_auth_token=HUGGINGFACE_TOKEN
            )

            if torch.cuda.is_available():
                print("🚀 Memindahkan model diarization ke GPU...")
                diarization_pipeline.to(torch.device("cuda"))
                embedding_pipeline.to(torch.device("cuda"))
                print("✅ Model diarization berhasil dijalankan di GPU!")
            else:
                print("⚠️ GPU tidak tersedia, menggunakan CPU.")

            print("✅ Model diarization dan embedding berhasil dimuat!")
        except Exception as e:
            print(f"❌ Gagal memuat model Pyannote: {e}")
            raise RuntimeError("Gagal memuat model Pyannote.")

    if whisper_model is None:
        try:
            print("📥 Memuat model Whisper...")
            whisper_model = whisper.load_model("medium").to(DEVICE)
            print(f"✅ Model Whisper berhasil dimuat di {DEVICE.upper()}!")
        except Exception as e:
            print(f"❌ Gagal memuat model Whisper: {e}")
            raise RuntimeError("Gagal memuat model Whisper.")


load_models()


def extract_embedding(audio_path, start, end):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        segment = waveform[:, int(start * sample_rate):int(end * sample_rate)]

        if segment.shape[1] < sample_rate:  # kurang dari 1 detik audio
            print(f"⚠️ Segmen {start:.2f}-{end:.2f} terlalu pendek untuk ekstraksi embedding.")
            return None

        # Padding jika terlalu pendek
        if segment.shape[1] < sample_rate * 1.5:
            pad_len = int(sample_rate * 1.5) - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, pad_len))

        embedding = embedding_pipeline(segment)
        return embedding.cpu().detach().numpy()

    except Exception as e:
        print(f"❌ Gagal mendapatkan embedding: {e}")
        return None



def apply_spectral_clustering(diarization_segments, n_speakers=2):
    embeddings = []
    valid_segments = []

    for seg in diarization_segments:
        if seg["embedding"] is not None:
            embeddings.append(seg["embedding"])
            valid_segments.append(seg)

    if len(embeddings) < n_speakers:
        print("⚠️ Jumlah embedding terlalu sedikit, clustering dilewati.")
        return diarization_segments

    clustering = SpectralClustering(
        n_clusters=n_speakers, assign_labels="discretize", random_state=42
    ).fit(embeddings)

    for i, seg in enumerate(valid_segments):
        seg["speaker"] = f"Speaker {clustering.labels_[i]}"

    return diarization_segments


def convert_to_wav(input_path, output_path):
    """Konversi file audio ke format WAV 16kHz mono"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("❌ ffmpeg tidak ditemukan! Pastikan ffmpeg terinstal.")

    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]
    subprocess.run(
        ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )


def play_audio_to_virtual_mic(audio_segment):
    try:
        # Di Linux, pastikan pakai virtual mic (contoh: pavucontrol + modprobe v4l2loopback)
        play(audio_segment)
    except Exception as e:
        print(f"❌ Gagal memutar audio: {e}")


def format_timestamp(seconds):
    """Mengubah detik menjadi format MM:SS"""
    return str(timedelta(seconds=int(seconds)))[2:]


def cleanup_files(*file_paths):
    """Menghapus file sementara setelah diproses"""
    time.sleep(1)  # Tunggu 1 detik sebelum menghapus file
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ File {file_path} berhasil dihapus!")
            else:
                print(f"⚠️ File {file_path} tidak ditemukan!")
        except Exception as e:
            print(f"⚠️ Gagal menghapus {file_path}: {e}")


@router.post("/transcribe")
async def transcribe_audio_api(file: UploadFile = File(...)):
    """API untuk hanya melakukan transkripsi dengan Whisper lokal"""
    unique_id = str(uuid.uuid4())
    temp_audio_mp3 = f"temp_{unique_id}.mp3"
    temp_audio_wav = f"temp_{unique_id}.wav"

    try:
        with open(temp_audio_mp3, "wb") as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)

        # 🔄 Konversi ke WAV
        convert_to_wav(temp_audio_mp3, temp_audio_wav)

        # 🔍 **Transkripsi dengan Whisper Lokal**
        whisper_response = whisper_model.transcribe(temp_audio_wav)

    except Exception as e:
        print(f"❌ Error terjadi sebelum penghapusan file: {e}")
        cleanup_files(temp_audio_mp3, temp_audio_wav)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        cleanup_files(temp_audio_mp3, temp_audio_wav)

    return {"transcription": whisper_response["text"]}


@router.get("/data")
async def get_user():
    return {"data": "jaya"}


# @router.post("/diarize")
# async def process_transcription_with_diarization(file: UploadFile = File(...)):
#     """API untuk melakukan transkripsi dan diarization"""
#     unique_id = str(uuid.uuid4())
#     temp_audio_mp3 = f"temp_{unique_id}.mp3"
#     temp_audio_wav = f"temp_{unique_id}.wav"

#     try:
#         with open(temp_audio_mp3, "wb") as temp_audio:
#             shutil.copyfileobj(file.file, temp_audio)

#         convert_to_wav(temp_audio_mp3, temp_audio_wav)

#         whisper_response = whisper_model.transcribe(
#             temp_audio_wav, word_timestamps=True
#         )
#         full_transcription = whisper_response["text"]
#         words = whisper_response["segments"]

#         diarization_result = diarization_pipeline(temp_audio_wav)

#         TOLERANCE = 0.10  # 200ms toleransi

#         diarization_segments = []
#         for diarized in diarization_result.itertracks(yield_label=True):
#             start_time, end_time, speaker = (
#                 diarized[0].start,
#                 diarized[0].end,
#                 diarized[2],
#             )

#             # Menggunakan toleransi untuk menangkap teks yang hampir masuk dalam segmen
#             text_segment = " ".join(
#                 word["text"]
#                 for word in words
#                 if (start_time - TOLERANCE) <= word["start"] <= (end_time + TOLERANCE)
#             )

#             embedding = extract_embedding(temp_audio_wav, start_time, end_time)

#             # Hanya tambahkan segmen jika ada teks dan durasi lebih dari 1 detik
#             if end_time - start_time > 1.0 and text_segment.strip():
#                 diarization_segments.append(
#                     {
#                         "start": format_timestamp(start_time),
#                         "end": format_timestamp(end_time),
#                         "speaker": speaker,
#                         "text": text_segment,
#                         "embedding": embedding if embedding is not None else None,
#                     }
#                 )

#         diarization_segments = apply_spectral_clustering(
#             diarization_segments, n_speakers=3
#         )

#         summarize = ringkas_transkrip(full_transcription)
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         # Menghapus file sementara jika ada
#         cleanup_files(temp_audio_mp3, temp_audio_wav)
#     return {
#         "transcription": full_transcription,
#         "summary": summarize,
#         "diarization": [
#             {
#                 "start": seg["start"],
#                 "end": seg["end"],
#                 "speaker": seg["speaker"],
#                 "text": seg["text"],
#             }
#             for seg in diarization_segments
#         ],
#     }

@router.get("/result/{uuid}")
async def get_result(uuid: str):
    response = supabase.table("transcriptions").select("*").eq("uuid", uuid).execute()
    data = response.data[0] if response.data else None

    if not data:
        raise HTTPException(status_code=404, detail="UUID not found")
    
    return data

@router.post("/diarize")
async def upload_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    unique_id = str(uuid.uuid4())
    temp_audio_mp3 = f"temp_{unique_id}.mp3"

    with open(temp_audio_mp3, "wb") as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)

    # Insert awal ke Supabase
    supabase.table("transcriptions").insert({"uuid": unique_id, "status": "processing"}).execute()

    # background_tasks.add_task(process_audio_job, unique_id, temp_audio_mp3)

    background_tasks.add_task(process_audio_job, unique_id, temp_audio_mp3)
    

    return {"uuid": unique_id}


def process_audio_job(uuid: str, temp_audio_mp3: str):
    temp_audio_wav = temp_audio_mp3.replace(".mp3", ".wav")

    try:
        # Step 1: Konversi ke WAV
        convert_to_wav(temp_audio_mp3, temp_audio_wav)

        # Step 2: Transkripsi dengan Whisper
        whisper_response = whisper_model.transcribe(
            temp_audio_wav, word_timestamps=True
        )
        full_transcription = whisper_response["text"]
        words = whisper_response["segments"]

        # Step 3: Diarization
        diarization_result = diarization_pipeline(temp_audio_wav)
        TOLERANCE = 0.10  # toleransi waktu
        diarization_segments = []

        for diarized in diarization_result.itertracks(yield_label=True):
            start_time, end_time, speaker = diarized[0].start, diarized[0].end, diarized[2]
            text_segment = " ".join(
                word["text"]
                for word in words
                if (start_time - TOLERANCE) <= word["start"] <= (end_time + TOLERANCE)
            )
            if end_time - start_time > 1.0 and text_segment.strip():
                diarization_segments.append({
                    "start": format_timestamp(start_time),
                    "end": format_timestamp(end_time),
                    "speaker": speaker,
                    "text": text_segment,
                    # Opsional simpan embedding
                    "embedding": extract_embedding(temp_audio_wav, start_time, end_time)
                })

        # Step 4: Clustering speaker jika dibutuhkan
        diarization_segments = apply_spectral_clustering(diarization_segments, n_speakers=3)

        # Step 5: Ringkasan teks
        summary = ringkas_transkrip(full_transcription)

        # Step 6: Simpan hasil ke Supabase
        supabase.table("transcriptions").update({
            "status": "done",
            "transcription": full_transcription,
            "summary": summary,
            "diarization": diarization_segments
        }).eq("uuid", uuid).execute()

    except Exception as e:
        # Simpan error ke Supabase jika gagal
        supabase.table("transcriptions").update({
            "status": "failed",
            "transcription": None,
            "summary": None,
            "diarization": None
        }).eq("uuid", uuid).execute()
        print("❌ Gagal memproses audio:", e)

    finally:
        # Hapus file sementara
        cleanup_files(temp_audio_mp3, temp_audio_wav)


@router.post("/join-meeting")
async def join_meeting_with_audio(
    meet_url: str = Form(...), audio_path: UploadFile = File(...)
):
    def run_bot():
        try:
            print("🎯 Memulai bot meeting...")

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=False,
                    args=[
                        "--use-fake-ui-for-media-stream",
                        "--use-fake-device-for-media-stream",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                context = browser.new_context()
                page = context.new_page()

                print(f"🌐 Membuka {meet_url}")
                page.goto(meet_url)

                time.sleep(10)  # Tunggu halaman siap, bisa disesuaikan

                print("🔊 Memainkan audio...")
                audio = AudioSegment.from_file(audio_path)
                play_audio_to_virtual_mic(audio)

                print("✅ Bot selesai join dan memainkan audio.")

                time.sleep(len(audio) / 1000 + 5)
                browser.close()

        except Exception as e:
            print(f"❌ Error bot meeting: {e}")

    # Jalankan bot di thread terpisah agar API tetap responsif
    thread = threading.Thread(target=run_bot)
    thread.start()

    return {"status": "Bot is joining meeting", "url": meet_url}
