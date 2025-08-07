# supabase_client.py
from supabase import create_client
import os
from dotenv import load_dotenv

# load_dotenv()
print("Available secrets:", os.listdir("/secrets"))
if os.path.exists("/secrets/ENV_VAR_NAME"):
    load_dotenv("/secrets/ENV_VAR_NAME")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)
