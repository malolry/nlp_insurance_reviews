# Review Prediction Application

This is a Streamlit web application that predicts the star rating and the main subject (theme) of an insurance review. It uses pretrained NLP models in the backend:

- A supervised model to predict the number of stars (1 to 5).
- A zero-shot classifier (using Facebook's Bart-large-MNLI) to predict the review theme (e.g., "Pricing", "Coverage", "Enrollment", "Customer Service", "Claims Processing", and "Cancellation").

## Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/malolry/nlp_insurance_reviews.git
   cd nlp_insurance_reviews

2. **Create a virtual environment and download packages:**

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. **Run the streamlit app:**

   ```bash
   streamlit run app.py


Enjoy this project :)
CHAIX Gabriel, LEROY Malo

