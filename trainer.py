import tensorflow as tf
from transformers import TFBertForSequenceClassification
from sklearn.metrics import classification_report



class Trainer:
    def __init__(self, options):
        self.opt = options

        # Load the BERT model for sequence classification
        self.model = TFBertForSequenceClassification.from_pretrained(self.opt.model_path, num_labels=3)

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.opt.learning_rate, epsilon=self.opt.epsilon)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Print the model summary
        self.model.summary()

    def train(self, X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded):
        # Train the model
        history = self.model.fit(
            {'input_ids': X_train_encoded['input_ids'], 'attention_mask': X_train_encoded['attention_mask']},
            y_train_encoded,
            validation_data=(
                {'input_ids': X_test_encoded['input_ids'], 'attention_mask': X_test_encoded['attention_mask']},
                y_test_encoded
            ),
            epochs=self.opt.num_epochs,
            batch_size=self.opt.batch_size,
        )

    def save_model(self):
        # Save the entire model
        self.model.save(self.opt.model_output)


    def evaluate(self, X_test_encoded, y_test_encoded):
        # Predict on the test set
        y_pred_prob = self.model.predict([X_test_encoded['input_ids'], X_test_encoded['attention_mask']])
        y_pred_class = y_pred_prob.logits.argmax(axis=-1)

        # Generate a classification report
        print(classification_report(y_test_encoded, y_pred_class))

 