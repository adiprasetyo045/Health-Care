import sys
import os

print("📂 Current Working Directory:", os.getcwd())
print("\n🐍 Python Sys Path:")
for p in sys.path:
    print(f" - {p}")

print("\n🔍 Trying to import Backend...")
try:
    import Backend
    print("✅ SUKSES: Backend package ditemukan!")
    print(f"   Lokasi: {Backend.__file__}")
except ImportError as e:
    print(f"❌ GAGAL: {e}")
    print("   Pastikan folder 'Backend' ada di Current Working Directory.")