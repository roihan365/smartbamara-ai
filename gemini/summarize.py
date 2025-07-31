import os
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import json

# Konfigurasi API key Gemini
genai.configure(api_key="AIzaSyAYqixBV5_DR_mG2h4ZC0s_zi5cNI53fa4")

# Konfigurasi model Gemini
generation_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Prompt dasar untuk merangkum transkrip
base_prompt = """
Saya memiliki transkrip hasil wawancara/meeting. Tolong buatkan ringkasan dalam **format JSON** dengan struktur berikut:

{
  "ringkasan_umum": "",
  "peserta": [],
  "topik_dan_pembahasan": "",
  "kesimpulan": "",
  "action_items": []
}

Gunakan bahasa Indonesia yang jelas. Berikut transkripnya:
"""

# Fungsi untuk merangkum transkrip
def ringkas_transkrip(transkrip):
    full_prompt = base_prompt + "\n" + transkrip

    response = model.generate_content([
        "input: " + full_prompt,
        "output:"
    ])

    try:
        # Parsing level pertama
        raw_response = json.loads(response.text)

        # Jika masih ada string JSON di dalamnya (seperti di bawah "response"), parse lagi
        if "response" in raw_response:
            return json.loads(raw_response["response"])
        else:
            return raw_response

    except Exception as e:
        print("⚠️ Parsing gagal:", e)
        print("↩️ Response mentah:", response.text)
        return {"raw": response.text}