from model import BertModel
from preprocessing import Preprocessing
import tensorflow as tf
from transformers import BertTokenizer
from random import randrange

# Utility function to round to next int
# Source: https://stackoverflow.com/questions/31818050/round-number-to-nearest-integer
def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])

# Convert the prediction to a result corresponding to the given scale
def prediction_to_result(prediction):
    scale = {0: "Not Funny", 1: "Slightly Funny", 2: "Moderately Funny", 3: "Funny"}
    rounded_pred = int(proper_round(prediction))
    if 3 >= rounded_pred >= 0:
        return scale[rounded_pred]
    else:
        return "Wrong input"

bert = BertModel()
# Load model
bert.bert_model.load_weights("BertModel/bert_model")

# Load random sentence from the test set
originalSent = bert.preprocessing.test_set["original"][randrange(len(bert.preprocessing.test_set["original"])-1)]
editedWord = bert.preprocessing.test_set["edit"][randrange(len(bert.preprocessing.test_set["edit"])-1)]
actual_grade = bert.preprocessing.test_set["meanGrade"][randrange(len(bert.preprocessing.test_set["meanGrade"])-1)]
# Setup test sentence for prediction
nOrig = originalSent.replace("<", "").replace("/>", "")
lEditSent = originalSent.split("<")
rEditSent = originalSent.split("/>")
nEdit = lEditSent[0] + editedWord + rEditSent[1]
input_sentence = nOrig + "[SEP]" + nEdit

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
tokenized_sent = tokenizer.encode_plus(input_sentence, max_length=256, padding="max_length",
                                           truncation=True, add_special_tokens=True, return_tensors="tf",return_token_type_ids=False)


input = {"input_ids": tf.cast(tokenized_sent["input_ids"], tf.int32),
         "attention_mask": tf.cast(tokenized_sent["attention_mask"], tf.int32)}
prediction = bert.bert_model.predict(input)[0][0][0]

# Output
print("\n______________________________________",
      "\nOriginal sentence: ", nOrig,
      "\nEdited sentence: ", nEdit,
      "\n______________________________________",
      "\nPrediction: ", prediction_to_result(prediction),
      "\nModel mean grade prediction: ", prediction,
      "\n______________________________________",
      "\nActual mean grade: ", actual_grade)
