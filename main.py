import sys
from data_loader  import load_data
from preprocessor import preprocess
from models       import train_all
from predictor    import launch_widget

def main():
    # loads and labels the raw dataset
    print("Loading data...")
    data_fake, data_true, data_fake_manual_testing, data_true_manual_testing = load_data()

    # cleans data and converts text to numerical vectors
    print("Preprocessing data...")
    xv_train, xv_test, y_train, y_test, vectorization = preprocess(data_fake, data_true)

    # trains multiple classifiers and outputs performance reports
    print("Starting training...")
    LR, DT, GB, RF = train_all(xv_train, xv_test, y_train, y_test, vectorization)

    # Check if we are in an interactive environment (like Jupyter)
    # If not, skip the widget as it won't work in terminal
    if hasattr(sys, 'ps1') or 'ipykernel' in sys.modules:
        print("Launching interactive widget...")
        launch_widget(vectorization, LR, DT, GB, RF)
    else:
        print("\nTraining complete. Models saved. Use 'streamlit run app.py' to start the web interface.")

if __name__ == "__main__":
    main()