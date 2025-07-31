# FastAPI Project

## 📌 Instalasi

Pastikan Anda memiliki **Python 3.8+** terinstal di sistem Anda.

1. **Clone repository** (jika belum dilakukan):
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Menjalankan Server

Gunakan perintah berikut untuk menjalankan aplikasi:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server akan berjalan di: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 📜 Dokumentasi API

Setelah server berjalan, Anda dapat mengakses dokumentasi API otomatis di:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## 🛠 Fitur
- Dibangun menggunakan **FastAPI** ⚡
- Mendukung **Swagger UI** & **ReDoc** untuk dokumentasi API 📜
- Mudah di-deploy dengan **Uvicorn** 🚀

Happy coding! 🚀

