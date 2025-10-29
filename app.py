import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cinema Theater Styling
st.set_page_config(
    page_title="🎭 CineMagic Theater",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [Keep all the same CSS styling from previous code...]

# Load movie data with debugging
@st.cache_data
def load_data():
    try:
        movies_df = pd.read_csv('movies.csv')
        st.sidebar.success(f"✅ Successfully loaded {len(movies_df)} movies!")
        st.sidebar.write(f"First 5 movies: {list(movies_df['title'].head())}")
        return movies_df
    except Exception as e:
        st.error(f"❌ Error loading movies: {e}")
        return pd.DataFrame()

movies_df = load_data()

# Show debug info
if not movies_df.empty:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🐛 Debug Info")
    st.sidebar.write(f"Total movies loaded: **{len(movies_df)}**")
    st.sidebar.write(f"Columns: {list(movies_df.columns)}")
    
    # Check for duplicates
    duplicates = movies_df['movieId'].duplicated().sum()
    if duplicates > 0:
        st.sidebar.warning(f"Duplicate movie IDs: {duplicates}")

# [Rest of your app code remains the same...]

# Theater Header
st.markdown("""
<div class="theater-header">
    <h1 style="color: #FFD700; font-size: 4rem; margin: 0; text-shadow: 3px 3px 5px rgba(0,0,0,0.5);">🎭 CINEMAGIC THEATER</h1>
    <p style="color: white; font-size: 1.5rem; margin: 10px 0 0 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">Your Personal Cinema Experience</p>
</div>
""", unsafe_allow_html=True)

# Show movie count prominently
if not movies_df.empty:
    st.markdown(f"""
    <div style="text-align: center; background: linear-gradient(90deg, #00008B 0%, #8B0000 100%); padding: 15px; border-radius: 10px; border: 2px solid #FFD700; margin: 20px 0;">
        <h2 style="color: #FFD700; margin: 0;">NOW SHOWING: {len(movies_df)} BLOCKBUSTER MOVIES!</h2>
    </div>
    """, unsafe_allow_html=True)

# [Rest of your tabs and functionality...]
