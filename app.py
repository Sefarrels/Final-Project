import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# ===============================
# Streamlit Theme: Netflix Style
# ===============================
st.set_page_config(
    page_title="Netflix Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for Netflix look
st.markdown(
    """
    <style>
    .chat-bubble-user {
        display: block;
        margin-left: auto;
        margin-right: 0;
        margin-bottom: 0.5rem;
        padding: 1rem;
        border-radius: 1rem 0 0 1rem;
        text-align: right;
        max-width: 70%;
        background: #e50914;
        color: #fff;
    }
    .chat-bubble-assistant {
        display: block;
        margin-right: auto;
        margin-left: 0;
        margin-bottom: 0.5rem;
        padding: 1rem;
        border-radius: 0 1rem 1rem 0;
        text-align: left;
        max-width: 70%;
        background: #fff;
        color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# ===============================
# 0. Data Structures & Error Classes
# ===============================
class APIError(Exception): pass
class DataLoadError(Exception): pass

@dataclass
class Movie:
    title: str
    genre: List[str]
    release_date: str
    vote_average: float
    vote_count: int
    popularity: float
    overview: str
    poster: Optional[str] = None

    @staticmethod
    def from_row(row: pd.Series) -> 'Movie':
        return Movie(
            title=row.get("Title", ""),
            genre=[g.strip() for g in str(row.get("Genre", "")).split(",") if g.strip()],
            release_date=row.get("Release_Date", ""),
            vote_average=float(row.get("Vote_Average", 0) or 0),
            vote_count=int(row.get("Vote_Count", 0) or 0),
            popularity=float(row.get("Popularity", 0) or 0),
            overview=row.get("Overview", "") if isinstance(row.get("Overview", ""), str) else "",
            poster=(row.get("Poster_Url", None).replace("w500", "w200") if isinstance(row.get("Poster_Url", None), str) and row.get("Poster_Url", None).strip() else None)
        )

class MovieRecommender:
    def __init__(self, movies: List[Movie]):
        self.movies = movies
        self.genre_map = self._build_genre_map()

    def _build_genre_map(self) -> Dict[str, List[Movie]]:
        genre_map = {}
        for m in self.movies:
            for g in m.genre:
                g_norm = g.lower().strip()
                genre_map.setdefault(g_norm, []).append(m)
        return genre_map

    def get_movies_for_genre(self, genre: str) -> List[Movie]:
        genre_norm = genre.lower().strip()
        movies = self.genre_map.get(genre_norm, [])
        return sorted(movies, key=lambda m: (m.vote_average, m.popularity), reverse=True)

    def all_genres(self) -> List[str]:
        return sorted(set(g for m in self.movies for g in m.genre if g))

class ChatSession:
    def __init__(self, title: str):
        self.title = title
        self.messages: List[Dict[str, Any]] = [
            {"role": "assistant", "content": "Tulis genre film kesukaanmu (contoh: Action, Comedy, Horror) 🎬"}
        ]
        self.recommended_all: List[Movie] = []
        self.recommend_index: int = 0

    def to_dict(self):
        return {
            "title": self.title,
            "messages": self.messages,
            "recommended_all": self.recommended_all,
            "recommend_index": self.recommend_index
        }

    @staticmethod
    def from_dict(d):
        cs = ChatSession(d["title"])
        cs.messages = d["messages"]
        cs.recommended_all = d.get("recommended_all", [])
        cs.recommend_index = d.get("recommend_index", 0)
        return cs

class Chatbot:
    def __init__(self, recommender: MovieRecommender):
        self.recommender = recommender

    def get_ai_response(self, messages_payload, api_key, restrict_to_dataset=True):
        if not api_key:
            return "⚠️ Masukkan API Key kalau mau tanya bebas ke AI."

        if restrict_to_dataset:
            movie_samples = self.recommender.movies[:20]
            system_prompt = {
                "role": "system",
                "content": (
                    "Kamu adalah asisten rekomendasi film Netflix. "
                    "Jawabanmu hanya berdasarkan film di dataset lokal yang tersedia."
                    + "\n\nContoh film:\n"
                    + "\n".join([f"- {m.title} ({', '.join(m.genre)})" for m in movie_samples])
                )
            }
        else:
            system_prompt = {
                "role": "system",
                "content": "Kamu adalah asisten AI yang ramah dan membantu. Jawab pertanyaan user sebaik mungkin."
            }

        allowed_roles = {"system", "assistant", "user", "function", "tool", "developer"}
        sanitized = [m for m in messages_payload if m.get("role", "") in allowed_roles]
        full_messages = [system_prompt] + sanitized

        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",
                    "messages": full_messages,
                    "max_tokens": 800,
                    "temperature": 0.7,
                }),
                timeout=20,
            )
        except Exception as e:
            raise APIError(f"API Error: {str(e)}")

        if response.status_code != 200:
            raise APIError(f"API Error: {response.text}")

        data = response.json()
        choice = data["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
        return "⚠️ Tidak ada respons dari AI"


# --- PATCH: Helper to ensure Movie object from dict if needed ---
def ensure_movie_list(movies_batch):
    # Accepts list of dict or Movie, returns list of Movie
    result = []
    for m in movies_batch:
        if isinstance(m, Movie):
            result.append(m)
        elif isinstance(m, dict):
            result.append(Movie(**m))
    return result


# ===============================
# 1. Fungsi AI (dengan sanitasi pesan)
# ===============================
def get_ai_response(messages_payload, api_key, df, restrict_to_dataset=True):
    """
    Memanggil OpenRouter (GPT). Jika restrict_to_dataset=True, AI hanya boleh jawab berdasarkan dataset lokal.
    Jika restrict_to_dataset=False, AI boleh jawab bebas.
    """
    if not api_key:
        return "⚠️ Masukkan API Key kalau mau tanya bebas ke AI."

    if restrict_to_dataset:
        # contoh film untuk system prompt (context)
        movie_samples = df[["Title", "Genre"]].dropna().head(20).to_dict(orient="records")
        system_prompt = {
            "role": "system",
            "content": (
                "Kamu adalah asisten rekomendasi film Netflix. "
                "Jawabanmu hanya berdasarkan film di dataset lokal yang tersedia."
                + "\n\nContoh film:\n"
                + "\n".join([f"- {m['Title']} ({m['Genre']})" for m in movie_samples])
            )
        }
    else:
        system_prompt = {
            "role": "system",
            "content": "Kamu adalah asisten AI yang ramah dan membantu. Jawab pertanyaan user sebaik mungkin."
        }

    # Sanitasi: hanya kirimkan role yang valid ke model
    allowed_roles = {"system", "assistant", "user", "function", "tool", "developer"}
    sanitized = []
    for m in messages_payload:
        role = m.get("role", "")
        if role in allowed_roles:
            sanitized.append(m)
        else:
            # skip non-supported roles like 'movies'
            continue

    full_messages = [system_prompt] + sanitized

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            data=json.dumps({
                "model": "openai/gpt-4o-mini",
                "messages": full_messages,
                "max_tokens": 800,
                "temperature": 0.7,
            }),
            timeout=20,
        )
    except Exception as e:
        return f"API Error: {str(e)}"

    if response.status_code != 200:
        # tampilkan body error agar gampang debug
        return f"API Error: {response.text}"

    data = response.json()
    choice = data["choices"][0]

    if "message" in choice and "content" in choice["message"]:
        return choice["message"]["content"]

    if "text" in choice:
        return choice["text"]

    return "⚠️ Tidak ada respons dari AI"


# ===============================
# 2. Load Dataset Netflix (OOP)
# ===============================
@st.cache_data
def load_movies() -> List[Movie]:
    import os
    base_path = os.path.dirname(__file__)
    csv_path = os.path.join(base_path, "netflix.csv")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin1", engine="python", on_bad_lines="skip")
    for col in ["Vote_Average", "Popularity", "Vote_Count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Vote_Average", "Popularity", "Vote_Count"], how="any")
    movies = [Movie.from_row(row) for _, row in df.iterrows()]
    return movies

movies = load_movies()
recommender = MovieRecommender(movies)
ALL_GENRES = recommender.all_genres()
chatbot = Chatbot(recommender)


# ===============================
# 3. Helper: tampilkan kartu film rapih (OOP)
# ===============================

def show_movie_recommendations(movies_batch: list):
    # Pastikan semua item adalah Movie object
    def ensure_movie_list(movies_batch):
        result = []
        for m in movies_batch:
            if isinstance(m, Movie):
                result.append(m)
            elif isinstance(m, dict):
                result.append(Movie(**m))
        return result

    movies_batch = ensure_movie_list(movies_batch)
    for m in movies_batch:
        with st.container():
            cols = st.columns([1, 4])
            if getattr(m, 'poster', None):
                try:
                    cols[0].image(m.poster, width=120)
                except Exception:
                    cols[0].write("")
            else:
                cols[0].write("")  # space placeholder

            title_line = f"### 🎬 {getattr(m, 'title', 'Unknown') or 'Unknown'} ({getattr(m, 'release_date', '')})"
            stats_line = f"⭐ **{getattr(m, 'vote_average', '-')}** | 👍 **{getattr(m, 'vote_count', '-')} votes** | 🔥 **{getattr(m, 'popularity', '-')}**"
            overview = getattr(m, 'overview', None) or "Tidak ada sinopsis."

            cols[1].markdown(title_line)
            cols[1].write(stats_line)
            cols[1].write(f"📖 {overview}")
        st.markdown("---")


# ===============================
# 5. Multi-Chat Session Init (OOP)
if "chats" not in st.session_state:
    st.session_state.chats = {}  # id -> ChatSession
    st.session_state.next_chat_id = 1
    st.session_state.active_chat_id = None

def start_new_chat(initial_prompt=None):
    chat_id = st.session_state.next_chat_id
    title = f"Chat {chat_id}"
    chat_obj = ChatSession(title)
    if initial_prompt:
        chat_obj.messages.append({"role": "user", "content": initial_prompt})
    st.session_state.chats[chat_id] = chat_obj
    st.session_state.active_chat_id = chat_id
    st.session_state.next_chat_id += 1
    return chat_id

# ===============================
# 6. Sidebar & Mode selector
# ===============================
st.sidebar.image("images/netflix_logo.png", width=120)
st.sidebar.title("⚙️ Pilih Mode Aplikasi")
mode = st.sidebar.radio("Mode", ["💬 Chatbot AI", "🎬 Rekomendasi Film"])

# Sidebar chat controls (only in Chatbot mode)
if mode == "💬 Chatbot AI":
    st.sidebar.subheader("💾 Chat Sessions")
    if st.sidebar.button("➕ Chat Baru"):
        start_new_chat()

    # daftar chat
    for cid, chat_meta in st.session_state.chats.items():
        label = chat_meta.title
        if st.sidebar.button(label, key=f"open_chat_{cid}"):
            st.session_state.active_chat_id = cid

    # jika belum ada chat sama sekali, buat satu
    if st.session_state.active_chat_id is None:
        start_new_chat()


# ===============================
# 7. Mode: Chatbot AI (Hybrid + pagination)
# ===============================
if mode == "💬 Chatbot AI":
    st.title("🤖 Netflix Movie Chatbot")

    # Simpan dan ambil API key dari session_state agar tidak hilang saat pindah mode
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    api_key = st.sidebar.text_input("🔑 Masukkan API Key OpenRouter (opsional)", type="password", value=st.session_state.api_key)
    st.session_state.api_key = api_key

    # get current chat object
    chat: ChatSession = st.session_state.chats[st.session_state.active_chat_id]

    # judul chat di header
    st.subheader(chat.title)

    # Render chat messages (inline). 'movies' role akan dirender oleh show_movie_recommendations.
    for msg in chat.messages:
        role = msg.get("role")
        if role == "user":
            st.markdown(
                f"<div class='chat-bubble-user'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        elif role == "assistant":
            st.markdown(
                f"<div class='chat-bubble-assistant'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        elif role == "movies":
            movies_batch = msg.get("content", [])
            # convert dict to Movie if needed
            if movies_batch and isinstance(movies_batch[0], dict):
                movies_batch = [Movie(**m) if not isinstance(m, Movie) else m for m in movies_batch]
            show_movie_recommendations(movies_batch)

    # Input
    if prompt := st.chat_input("Ketik genre atau pertanyaan tentang film"):
        chat.messages.append({"role": "user", "content": prompt})
        if chat.title.startswith("Chat"):
            trimmed = prompt.strip()
            chat.title = (trimmed[:30] + ("..." if len(trimmed) > 30 else ""))
        normalized = prompt.strip().lower()
        next_keywords = {"mau", "lagi", "iya", "lanjut", "lanjutkan", "show more", "more", "next", "tampilkan lagi"}
        if normalized in next_keywords:
            if chat.recommended_all and chat.recommend_index < len(chat.recommended_all):
                idx = chat.recommend_index
                batch = chat.recommended_all[idx: idx + 3]
                if batch:
                    chat.messages.append({"role": "assistant", "content": "Oke, ini beberapa lagi:"})
                    chat.messages.append({"role": "movies", "content": [asdict(m) for m in batch]})
                    chat.recommend_index = idx + len(batch)
                    if chat.recommend_index < len(chat.recommended_all):
                        chat.messages.append({"role": "assistant", "content": "Mau aku carikan pilihan lainnya?"})
                    else:
                        chat.messages.append({"role": "assistant", "content": "Itu saja rekomendasi yang tersedia untuk genre tersebut."})
                else:
                    chat.messages.append({"role": "assistant", "content": "Tidak ada tambahan rekomendasi."})
            else:
                chat.messages.append({"role": "assistant", "content": "Belum ada rekomendasi sebelumnya. Ketik nama genre (mis. Action) untuk mulai."})
        else:
            genre_choice = None
            for g in ALL_GENRES:
                if normalized == g.lower():
                    genre_choice = g
                    break
            if genre_choice:
                all_movies = recommender.get_movies_for_genre(genre_choice)
                if all_movies:
                    chat.recommended_all = all_movies
                    chat.recommend_index = 0
                    batch = all_movies[0:3]
                    chat.recommend_index = len(batch)
                    chat.messages.append({"role": "assistant", "content": f"Aku temukan beberapa film genre **{genre_choice.title()}** buat kamu:"})
                    chat.messages.append({"role": "movies", "content": [asdict(m) for m in batch]})
                    if chat.recommend_index < len(chat.recommended_all):
                        chat.messages.append({"role": "assistant", "content": "Itu rekomendasi terbaik 🎥. Mau aku carikan pilihan lainnya?"})
                    else:
                        chat.messages.append({"role": "assistant", "content": "Itu saja rekomendasi yang tersedia untuk genre tersebut."})
                else:
                    chat.messages.append({"role": "assistant", "content": f"Maaf, tidak ada film dengan genre **{genre_choice.title()}**."})
            else:
                if api_key:
                    with st.spinner("🤖 AI sedang berpikir..."):
                        try:
                            ai_response = chatbot.get_ai_response(chat.messages, api_key, restrict_to_dataset=False)
                        except APIError as e:
                            ai_response = str(e)
                    chat.messages.append({"role": "assistant", "content": ai_response})
                else:
                    chat.messages.append({"role": "assistant", "content": "⚠️ Aku tidak mengenali genre itu. Masukkan API Key jika mau tanya bebas ke AI."})
        st.rerun()


# ===============================
# 8. Mode: Manual Rekomendasi (filter)
# ===============================
elif mode == "🎬 Rekomendasi Film":
    st.title("🎥 Netflix Movie Recommender (Filter Manual)")

    st.sidebar.header("🔎 Filter Rekomendasi Film")
    all_genres = ALL_GENRES
    selected_genre = st.sidebar.selectbox("🎭 Pilih genre", ["All"] + all_genres)
    search_query = st.sidebar.text_input("🔍 Cari judul/keyword")
    num_movies = st.sidebar.slider("Jumlah film yang ditampilkan", 1, 20, 5)

    filtered_movies = list(movies)
    if selected_genre != "All":
        filtered_movies = [m for m in filtered_movies if selected_genre in m.genre]
    if search_query:
        filtered_movies = [m for m in filtered_movies if search_query.lower() in m.title.lower()]

    st.sidebar.write(f"📊 Ditemukan {len(filtered_movies)} film")

    st.subheader("Rekomendasi Film")
    if filtered_movies:
        for m in filtered_movies[:num_movies]:
            with st.container():
                cols = st.columns([1, 4])
                if m.poster:
                    try:
                        cols[0].image(m.poster.replace("w500", "w150"), width=120)
                    except Exception:
                        cols[0].write("")
                else:
                    cols[0].write("")
                cols[1].markdown(f"### {m.title} ({m.release_date})")
                cols[1].write(f"⭐ {m.vote_average} | 👍 {m.vote_count} votes | 🔥 Popularity: {m.popularity}")
                cols[1].write(f"**Overview:** {m.overview}")
            st.markdown("---")
    else:
        st.info("Tidak ada film yang cocok dengan filter.")
