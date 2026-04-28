# SiMiJu — Sistem Deteksi Kemiripan Judul Skripsi
### STAB Negeri Raden Wijaya Wonogiri

## Struktur Folder

```
skripsi-app/
├── main.py                  ← Backend FastAPI + Uvicorn
├── requirements.txt
├── skripsi_dataset.csv      ← Dataset judul skripsi
└── static/
    └── index.html           ← Frontend website
```

## Setup & Jalankan

```bash
# 1. Install library
pip install -r requirements.txt

# 2. Development (auto-reload saat file berubah)
uvicorn main:app --reload --port 5000

# 3. Production (multi-worker)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

Buka browser: http://localhost:5000

## API Docs (otomatis dari FastAPI)
http://localhost:5000/api/docs

## Update Dataset
Ganti file skripsi_dataset.csv, lalu restart uvicorn.

## Deploy ke server/VPS
```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan dengan workers sesuai jumlah CPU
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# Atau gunakan systemd / supervisor agar berjalan terus
```
