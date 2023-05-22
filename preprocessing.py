import numpy as np
import pandas as pd
from transformers import BertTokenizer

"""
Utility function 
"""
def prep_map(sent_ids, masks, labels):
    return {"input_ids": sent_ids, "attention_mask": masks}, labels

"""
Class to preprocess data for learning/validation/testing
"""
class Preprocessing:

    """
    Loads, prepares and makes all datasets easily accessible
    """
    def __init__(self, max_seq_length=256, cap_data_size=False, data_set_size=50):
        '''
        :param max_seq_length: max length for sequence
        :param cap_data_size: True if you want to cap the data
        :param data_set_size: max data size to read
        '''
        self.__save_possible_gradings = True
        path_train = "datasets/train.csv"
        path_val = "datasets/dev.csv"
        path_test = "datasets/test.csv"
        # Read train to dataframe
        if cap_data_size:
            train_set = pd.read_csv(path_train)[:data_set_size]
            val_set = pd.read_csv(path_val)[:data_set_size]
            self.test_set = pd.read_csv(path_test)[:data_set_size]
        else:
            train_set = pd.read_csv(path_train)
            val_set = pd.read_csv(path_val)
            self.test_set = pd.read_csv(path_test)

        self.max_seq_len = max_seq_length
        self.train_dataset = self.__prepare_dataset(train_set, self.__save_possible_gradings)
        self.__save_possible_gradings = False
        self.val_dataset = self.__prepare_dataset(val_set, self.__save_possible_gradings)
        self.test_dataset = self.__prepare_dataset(self.test_set, self.__save_possible_gradings)

    """
    Prepare training data to the following input format:
    OriginalSentence[SEP]EditedSentence
    """
    def __prepare_sentences(self, orig, edit_words):
        ret = []
        i = 0
        while i < len(orig):
            nOrig = orig[i].replace("<", "").replace("/>", "")
            lEditSent = orig[i].split("<")
            rEditSent = orig[i].split("/>")
            nEdit = lEditSent[0] + edit_words[i] + rEditSent[1]
            ret.append(nOrig + "[SEP]" + nEdit)
            i += 1
        return ret

    """
    prepares and returns the datasets used for train/validate/test
    """
    def __prepare_dataset(self, data_set, save_possible_gradings):
        samples_count = len(data_set)

        # Process training data to input format
        processed_train_sentences = self.__prepare_sentences(data_set["original"], data_set["edit"])
        # Prepare labels
        gradings = data_set["meanGrade"].values
        possible_gradings = []

        for i in range(len(gradings)):
            if gradings[i] not in possible_gradings:
                possible_gradings.append(gradings[i])

        if save_possible_gradings:
            self.num_possible_gradings = len(possible_gradings)

        sent_ids = np.zeros((samples_count, self.max_seq_len))
        masks = np.zeros((samples_count, self.max_seq_len))

        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for i, sentence in enumerate(processed_train_sentences):
            tokens = tokenizer.encode_plus(sentence, max_length=self.max_seq_len, padding="max_length",
                                           truncation=True, add_special_tokens=True, return_tensors="tf")
            sent_ids[i, :] = tokens["input_ids"]
            masks[i, :] = tokens["attention_mask"]

        train_dataset = (dict(input_ids=sent_ids.astype("int32"), attention_mask=masks.astype("int32")), gradings)

        return train_dataset
