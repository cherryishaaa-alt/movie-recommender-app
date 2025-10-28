import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Make it look nice
st.set_page_config(
    page_title="MoviePal 🎬",
    page_icon="🎬",
    layout="centered"
)

# App title
st.title("🎬 MoviePal")
st.markdown("### Your Personal Movie Recommendation App")
st.write("Discover movies you'll love in seconds!")

# Load movie data
@st.cache_data
def load_data():
    return pd.read_csv('movies.csv')

movies_df = load_data()

# Recommendation function
def get_recommendations(movie_title, movies_df, num_recommendations=5):
    # Combine genres and description
    movies_df['content'] = movies_df['genres'] + ' ' + movies_df['description']
    
    # Create TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])
    
    # Calculate similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get movie index
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    return movies_df.iloc[movie_indices]

# Main app
st.header("🔍 Find Similar Movies")

selected_movie = st.selectbox(
    "Choose a movie you like:",
    movies_df['title'].tolist()
)

if st.button("🎯 Get Recommendations", type="primary"):
    with st.spinner('Finding your perfect movies...'):
        recommendations = get_recommendations(selected_movie, movies_df)
        
    st.success(f"Because you liked **{selected_movie}**, you might enjoy:")
    
    for i, row in recommendations.iterrows():
        with st.container():
            st.markdown(f"""
            **🎭 {row['title']}**
            - **Genre:** {row['genres']}
            - **About:** {row['description']}
            """)
            st.divider()

# Show all movies
st.header("📚 All Movies")
for i, row in movies_df.iterrows():
    st.write(f"**{row['title']}** - *{row['genres']}*")

st.markdown("---")
st.markdown("Built with ❤️ using AI")
