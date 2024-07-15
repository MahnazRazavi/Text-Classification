from transformers import BertTokenizer

class Token:
    def __init__(self) -> None: 
        # Load the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('./model/bert-base-uncased')

    # Tokenize the text
    def encode_texts(self, texts, max_length=128):
        return self.tokenizer.batch_encode_plus(
            texts.tolist(),
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )