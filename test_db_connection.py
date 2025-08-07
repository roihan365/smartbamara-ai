import os
import psycopg2
from dotenv import load_dotenv

# Load .env
if os.path.exists("/secrets"):
    print("✅ Secrets directory found.")
    print("Available secrets:", os.listdir("/secrets"))

    if os.path.exists("/secrets/ENV_VAR_NAME"):
        load_dotenv("/secrets/ENV_VAR_NAME")
    else:
        print("⚠️ Secret file '/secrets/ENV_VAR_NAME' not found.")
else:
    print("⚠️ Secrets directory '/secrets' not found. Probably running outside of Cloud Run Gen2.")

try:
    conn = psycopg2.connect(
        dbname=os.getenv("dbname"),
        user=os.getenv("user"),
        password=os.getenv("password"),
        host=os.getenv("host"),
        port=os.getenv("port"),
    )
    print("Koneksi ke database berhasil!")
    conn.close()
except Exception as e:
    print("Gagal koneksi ke database:", e)