import streamlit as st
from transformers import pipeline
import time
import joblib


# -----------------------------------------------------------------------------
# Prediction Functions
# -----------------------------------------------------------------------------

star_model = None

def load_star_model():
    global star_model # for caching the model
    if star_model is None:
        star_model = joblib.load("models/star_prediction_uni.pkl")
    return star_model

def predict_star_rating(review_text):
    """
    Predicts the star rating using a pre-trained model loaded from models/star_prediction.pkl.
    Returns:
        int: The predicted star rating (assumed to be an integer between 1 and 5).
    """    
    # Load the model 
    model = load_star_model()
    prediction = model.predict([review_text])
    
    # Return the first prediction
    return prediction[0]

def predict_theme(review_text, candidate_labels):
    """
    Zero-shot classifier to predict the theme of the review.
    Here we use Hugging Face's pipeline with "facebook/bart-large-mnli" model.
    """
    theme_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    result = theme_classifier(review_text, candidate_labels)
    
    # Return topp label and its score
    top_label = result["labels"][0]
    top_score = result["scores"][0]
    return top_label, top_score

# -----------------------------------------------------------------------------
# Streamlit app / Frontend
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Review Prediction Application",
    page_icon="⭐",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("Review Prediction Application")
st.write("Enter your review below to get a star rating prediction and a theme assignment.")

# Sidebar explanation 
st.sidebar.header("About")
st.sidebar.write(
    """
    This application predicts:
    - The number of stars (1 to 5) a review would receive.
    - The main subject (theme) of the review such as **Pricing**, **Coverage**, **Enrollment**, **Customer Service**, **Claims Processing**, or **Cancellation**.
    
    The predictions are made in real-time using pretrained NLP models.
    """
)

# Input: Text area for review
review_text = st.text_area("Enter your review:", height=200)

# Potential themes for the reviews to belong to
candidate_labels = ["Pricing", "Coverage", "Enrollment", "Customer Service", "Claims Processing", "Cancellation"]

# Button to trigger prediction
if st.button("Predict"):
    if review_text.strip() == "":
        st.error("Please enter a review before predicting!")
    else:
        # Show a spinner while predictions are being made
        with st.spinner("Making predictions..."):
            star_rating = predict_star_rating(review_text)
            theme, theme_score = predict_theme(review_text, candidate_labels)
            
            time.sleep(0.5) # Simulate processing
        
        st.success("Predictions complete!")
        st.markdown("### Prediction Results")
        st.write(f"**Predicted Star Rating:** {star_rating} ⭐")
        st.write(f"**Predicted Theme:** {theme} (Confidence: {theme_score:.2f})")
