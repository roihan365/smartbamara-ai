# supabase_client.py
from supabase import create_client
import os
from dotenv import load_dotenv

# load_dotenv()
if os.path.exists("/secrets/env-vars/.env"):
    load_dotenv("/secrets/env-vars/.env")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)
