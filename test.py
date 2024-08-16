import time
import numpy as np
from options import TextOptions
from tokenization import Token
import tensorflow as tf

class TextClassificatin:
    def __init__(self, options):
        self.opt = options
        # Load the trained model
        self.model = tf.saved_model.load(self.opt.model_output)

    def preprocessing(self, text):
        # Preprocessing: Convert text to lowercase and remove punctuation
        text = text.lower().replace('[^\w\s]', '')
        return text
    
    def text_classification(self, text):
        text = self.preprocessing(text)
        text_encode = Token(self.opt).encode_text(text)
        result = self.predict(text_encode)
        return result
    
    def predict(self, inputs):
        # Step 6: Get the default signature function for inference
        infer = self.model.signatures["serving_default"]

        # Step 7: Perform inference
        # Ensure the input tensor names match the expected inputs of the model
        outputs = infer(input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"], 
                        token_type_ids=inputs.get("token_type_ids"))

        # Step 8: Extract the logits and perform post-processing
        logits = outputs['logits']
        predicted_class = tf.argmax(logits, axis=1).numpy()

        return predicted_class

    # Function to measure inference time
    def measure_inference_time_bert(self, text, n_runs=10):
        inference_times = []
        start_time = time.time()
        for _ in range(n_runs):  # run multiple times to average the inference time
            result = self.text_classification(text)
            end_time = time.time()
            avg_inference_time = (end_time - start_time) / n_runs
            inference_times.append(avg_inference_time)
        return np.mean(inference_times), np.std(inference_times)
    

if __name__ == "__main__":
    options = TextOptions()
    opts = options.parse()
    tcl = TextClassificatin(opts)
    # Sample texts for inference time measurement
    sample_texts = "This is a test sentence."

    result = tcl.text_classification(sample_texts)

    # Measure inference time
    mean_time, std_time = tcl.measure_inference_time_bert(sample_texts)
    print(f"Average Inference Time: {mean_time:.6f} seconds")
    print(f"Standard Deviation of Inference Time: {std_time:.6f} seconds")