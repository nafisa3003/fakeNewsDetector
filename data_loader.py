import pandas as pd

def load_data():
    # loads fake and true news datasets and labels them
    data_fake = pd.read_csv('Fake.csv')
    data_true = pd.read_csv('True.csv')

    data_fake["class"] = 0
    data_true["class"] = 1

    # separates the last 10 rows of each dataset for manual evaluation
    data_fake_manual_testing = data_fake.tail(10).copy()
    data_true_manual_testing = data_true.tail(10).copy()

    # remove the manual testing rows from the training/test pool
    data_fake = data_fake.iloc[:-10]
    data_true = data_true.iloc[:-10]

    print(f"Fake news shape: {data_fake.shape}, True news shape: {data_true.shape}")

    return data_fake, data_true, data_fake_manual_testing, data_true_manual_testing