# setup_hf_login.py

from huggingface_hub import login

print("🔐 Hugging Face Login")
print("👉 Please paste your Hugging Face access token below.")
print("You can get it from https://huggingface.co/settings/tokens")

login()
print(" Login successful! You won't need to log in again on this machine.")
