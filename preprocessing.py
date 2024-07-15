import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset():
    # Load the dataset
    data = pd.read_csv('./dataset/xenophobia_racism_dataset.csv')
    # Display the first few rows
    print(data.head())
    print(data.shape)
    return data

def preprocessing(data):
    # Preprocessing: Convert text to lowercase and remove punctuation
    data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '')
    return data

def split_dataset(data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test