# supabase_client.py
from supabase import create_client
import os
from dotenv import load_dotenv

# Only try to load dotenv if /secrets exists
load_dotenv()
# if os.path.exists("/secrets"):
#     print("✅ Secrets directory found.")
#     print("Available secrets:", os.listdir("/secrets"))

#     if os.path.exists("/secrets/ENV_VAR_NAME"):
#         load_dotenv("/secrets/ENV_VAR_NAME")
#     else:
#         print("⚠️ Secret file '/secrets/ENV_VAR_NAME' not found.")
# else:
#     print("⚠️ Secrets directory '/secrets' not found. Probably running outside of Cloud Run Gen2.")

# Get variables (will be None if not loaded)
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set!")

supabase = create_client(supabase_url, supabase_key)