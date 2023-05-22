import tensorflow as tf
from preprocessing import Preprocessing
from transformers import TFBertForSequenceClassification

"""
One Model class in order to make initialisation easy with the prepared data
"""
class BertModel:
    def __init__(self, train_bert=True, data_set_size=50, cap_data_size=False):
        self.preprocessing = Preprocessing(data_set_size=data_set_size)

        # Model
        self.bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)
        self.bert_model.summary()
        # Settings
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        loss_function = tf.keras.losses.MeanSquaredError()
        accuracy = tf.keras.metrics.RootMeanSquaredError("Accuracy")
        self.bert_model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
