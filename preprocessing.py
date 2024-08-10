import pandas as pd
from sklearn.model_selection import train_test_split
from options import TextOptions

options = TextOptions()
opts = options.parse()

class Preprocessing:
    def load_dataset(self, opts):
        # Load the dataset
        data = pd.read_csv(opts.dataset)
        # Display the first few rows
        print(data.head())
        print(data.shape)
        return data

    def preprocessing(self, data):
        # Preprocessing: Convert text to lowercase and remove punctuation
        data['text'] = data['text'].str.lower().str.replace('[^\w\s]', '')
        return data

    def split_dataset(self, data):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def main(self):
        data = self.load_dataset(opts)
        pre_data = self.preprocessing(data)
        X_train, X_test, y_train, y_test = self.split_dataset(pre_data)
        return X_train, X_test, y_train, y_test
