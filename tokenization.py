from transformers import BertTokenizer
import pickle


class Token:
    def __init__(self): 
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("model/bert-base-uncased")

    # Tokenize the text
    def encode_texts(self, texts, max_length=128):
        return self.tokenizer.batch_encode_plus(
            texts.tolist(),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
    def encode_text(self, text, max_length=128):
        return self.tokenizer.encode_plus(
                text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='tf'
            )
    def main(self, X_train, X_test):
        X_train_encoded = self.encode_texts(X_train)
        X_test_encoded = self.encode_texts(X_test)
        return X_train_encoded, X_test_encoded

    def save(self, opts, X_train_encoded, X_test_encoded):
        # Save encoded data using pickle
        with open(opts.token_output + 'X_train_encoded.pkl', 'wb') as f:
            pickle.dump(X_train_encoded, f)

        with open(opts.token_output + 'X_test_encoded.pkl', 'wb') as f:
            pickle.dump(X_test_encoded, f)