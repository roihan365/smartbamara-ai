from playwright.sync_api import sync_playwright


def manual_google_login():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        page = context.new_page()
        print("🌐 Membuka halaman login Google...")

        # Buka halaman login Google
        page.goto("https://accounts.google.com/")

        # Tunggu kamu login manual
        input("✅ Selesai login akun Google dummy? Tekan ENTER...")

        # Simpan session ke auth.json
        context.storage_state(path="auth.json")
        print("💾 Session disimpan di 'auth.json'")

        browser.close()


manual_google_login()
