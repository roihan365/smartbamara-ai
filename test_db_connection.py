import os
import psycopg2
from dotenv import load_dotenv

# Load .env
load_dotenv()

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