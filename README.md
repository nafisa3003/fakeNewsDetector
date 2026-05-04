# Verity — Fake News Detector

A modular NLP application that uses four machine learning models (Logistic Regression, Decision Tree, Gradient Boosting, and Random Forest) to detect fake news articles via a majority-vote consensus.

## 🚀 Setup Instructions

1. **Clone the repository** to your local machine.
2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

### 1. Train the Models
Before running the app, you need to train the models and generate the saved files. This only needs to be done once:
```bash
python main.py
```

### 2. Launch the Web App
Once the models are saved (look for `.pkl` files), start the Streamlit interface:
```bash
streamlit run app.py
```

## 📂 Project Structure
- `app.py`: Streamlit web interface.
- `main.py`: Training pipeline entry point.
- `models.py`: Model training and persistence logic.
- `preprocessor.py`: Text cleaning and vectorization.
- `data_loader.py`: Data ingestion and splitting.
- `predictor.py`: Majority vote consensus logic.
