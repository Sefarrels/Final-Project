# Netflix Movie Recommender & Chatbot

Aplikasi Streamlit untuk merekomendasikan film Netflix dan chat dengan AI seputar film. Sudah OOP, UI modern, dan mudah dikustomisasi.

---

## Fitur Utama
- **Rekomendasi film** berdasarkan genre, judul, dan popularitas
- **Chatbot AI** (OpenRouter/GPT) untuk tanya jawab seputar film
- **Multi-chat session**: simpan riwayat chat
- **UI dark mode** dengan tema Netflix
- **Bubble chat** kanan (user) & kiri (AI)
- **Logo Netflix di sidebar kiri atas**
- **Input, dropdown, dan chatbox kontras, mudah dibaca**

---

## Cara Menjalankan
1. **Clone/download** repo/project ini
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```
4. **(Opsional) Atur theme**
   - Sudah otomatis dark mode via `.streamlit/config.toml`

---

## Struktur Folder
```
├── app.py                  # Main Streamlit app
├── netflix.csv             # Dataset film Netflix
├── requirements.txt        # Daftar dependencies
├── images/
│   └── netflix_logo.png    # Logo Netflix untuk sidebar
├── .streamlit/
│   └── config.toml         # Theme Streamlit
```

---

## Penggunaan
- **Sidebar**:
  - Pilih mode: Chatbot AI / Rekomendasi Film
  - Masukkan API Key (OpenRouter) jika ingin tanya bebas ke AI
  - Filter genre, cari judul, atur jumlah film
  - Logo Netflix tampil di kiri atas
- **Main Area**:
  - Chatbot: bubble chat kanan (user, merah) & kiri (AI, putih)
  - Rekomendasi: daftar film dengan poster, rating, overview

---

## LINK DEPLOY
https://netflixfilmrecommendation.streamlit.app/

## API Key
- Untuk fitur AI bebas, daftar gratis di [OpenRouter](https://openrouter.ai/)
- Masukkan API Key di sidebar

---


## Credits
- Dibuat dengan Python, Streamlit, dan OpenRouter API
- UI terinspirasi Netflix
- Dataset diambil dari [Kaggle Netflix Movies and TV Shows](https://www.kaggle.com/datasets/rahulverma07/netflix-movie-dataset) karya Rahul Verma, yang menyediakan data film dan serial Netflix secara komprehensif dan terkurasi.

---

## Lisensi
Bebas digunakan untuk belajar/non-komersial.
