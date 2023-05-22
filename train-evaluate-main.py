from model import BertModel
import sys

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("No valid option")

    else:
        bert = BertModel()

        if sys.argv[1] == "-trainModel":

            # Training
            bert_fit = bert.bert_model.fit(x=bert.preprocessing.train_dataset[0],
                                           y=bert.preprocessing.train_dataset[1],
                                           validation_data=bert.preprocessing.val_dataset,
                                           epochs=3)
            # Save model
            bert.bert_model.save_weights('BertModel/bert_model')
            # Evaluate
            bert.bert_model.evaluate(x=bert.preprocessing.test_dataset[0], y=bert.preprocessing.test_dataset[1])
        elif sys.argv[1] == "-loadModel":
            # Load model and evaluate
            bert.bert_model.load_weights("BertModel/bert_model")
            bert.bert_model.evaluate(x=bert.preprocessing.test_dataset[0], y=bert.preprocessing.test_dataset[1])
