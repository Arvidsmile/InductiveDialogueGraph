import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score
from sklearn.preprocessing import MultiLabelBinarizer
import sys
import argparse
import numpy as np
import pickle
import pandas as pd


def make_model(num_classes, learning_rate, metrics, activation = "softmax", loss = 'categorical_crossentropy'):
    model = Sequential()
    model.add(Dense(512, activation = 'relu', input_shape = (1024 , )))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = activation))
    # model.summary()
    model.compile(loss = loss,
                  optimizer = Adam(learning_rate = learning_rate),
                  metrics = metrics)
    return model

def createMultiLabels(dataset):

    labelsets = []
    # Loop through all of dataset['Dialogue Act']
    # and create a set (e.g. {'Greeting', 'Follow-Up Question'})
    # for each utterance. Then fit a MultiLabelBinarizer from sklearn
    # library and return the numpy array of multi-hot encoded vectors
    for DA in dataset['Dialogue Act']:
        labels = set(DA.split('-'))
        labelsets.append(labels)

    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(labelsets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help = "Which dataset do you want to generate ELMo embeddings for?",
                        type = str,
                        default = "SwDA")
    parser.add_argument("--testing",
                        help = "Are we in testing mode?",
                        type = int,
                        default = 0)
    parser.add_argument("--batch_size",
                        help = "Size of each batch of utterances.",
                        type = int,
                        default = 32)
    parser.add_argument("--epochs",
                        help = "Number of times to go through all batches",
                        type = int,
                        default = 50)
    parser.add_argument("--kfold",
                        help="Number of folds to do training/testing in",
                        type=int,
                        default=10)

    args = parser.parse_args()

    # Check which dataset we work on
    if args.dataset == 'SwDA':
        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")
        # Make labels into integer format
        dataset["Labels"] = pd.factorize(dataset["Dialogue Act"])[0]
        print(embeddings.shape)
        print(dataset["Labels"], dataset["Labels"].nunique())

    elif args.dataset == 'MRDA':
        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")
        # Make labels into integer format
        dataset["Labels"] = pd.factorize(dataset["Dialogue Act"])[0]
        print(embeddings.shape)
        print(dataset["Labels"], dataset["Labels"].nunique())

    elif args.dataset == 'MANtIS':
        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")

        # Create label-matrix with multi-hot encoded labels
        multi_hot_labels = createMultiLabels(dataset)

    elif args.dataset == 'MSDialog':
        # Load the embeddings
        embFile = open(f"../../ELMoPickled/ELMo_{args.dataset}.pickle", "rb")
        embeddings = pickle.load(embFile)
        # Load the original dataset
        dataset = pd.read_csv(f"../../CSVData/{args.dataset}.csv")

        # Create label-matrix with multi-hot encoded labels
        multi_hot_labels = createMultiLabels(dataset)

    else:
        print("Please enter a valid dataset: SwDA | MRDA | MANtIS | MSDialog | testing")
        sys.exit()

    # Report results and best model across
    fold_accuracies = []
    fold_microf1 = []
    fold_macrof1 = []
    fold_precision = []
    fold_losses = []

    fold = 0
    if args.dataset in ['SwDA', 'MRDA']:
        NUM_CLASSES = dataset["Labels"].nunique()
        labels = dataset["Labels"]
    else:
        NUM_CLASSES = multi_hot_labels.shape[1]
        labels = multi_hot_labels
    kfold = KFold(n_splits = args.kfold, shuffle = True)
    for train, test in kfold.split(embeddings, labels):

        if args.dataset in ['SwDA', 'MRDA']:
            model = make_model(NUM_CLASSES, 0.01, ['accuracy'])
            # One-hot encode the labels
            y_train = keras.utils.to_categorical(labels[train], NUM_CLASSES)
            y_test = keras.utils.to_categorical(labels[test], NUM_CLASSES)
        else:
            model = make_model(NUM_CLASSES, 0.01, ['accuracy'],
                               activation = 'sigmoid',
                               loss = 'binary_crossentropy')
            y_train = labels[train]
            y_test = labels[test]

        print("-" * 100, f"\nTraining Fold: {fold}\n")
        history = model.fit(embeddings[train],
                            y_train,
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            verbose=1
                            )

        score = model.evaluate(embeddings[test], y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        fold_accuracies.append(score[1])
        fold_losses.append(score[0])

        # Calculate microf1, macrof1 and precision
        prediction = model.predict(embeddings[test], verbose = 1)

        if args.dataset in ['SwDA', 'MRDA']:
            y_pred_bool = np.argmax(prediction, axis = 1)

            # Return y_test to single-digit labels
            y_test = np.argmax(y_test, axis = 1)
        else:
            # For binary crossentropy we use round to achieve a decision threshold at 0.5
            y_pred_bool = np.round(prediction)

        f1_micro = f1_score(y_test, y_pred_bool, average = 'micro')
        f1_macro = f1_score(y_test, y_pred_bool, average='macro')
        precision = precision_score(y_test, y_pred_bool, average = 'micro')

        fold_microf1.append(f1_micro)
        fold_macrof1.append(f1_macro)
        fold_precision.append(precision)

        fold += 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(fold_accuracies)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {fold_losses[i]} - Accuracy: {fold_accuracies[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(fold_accuracies)} (+- {np.std(fold_accuracies)})')
    print(f'> Precision: {np.mean(fold_precision)} (+- {np.std(fold_precision)})')
    print(f'> F1-Micro: {np.mean(fold_microf1)} (+- {np.std(fold_microf1)})')
    print(f'> F1-Macro: {np.mean(fold_macrof1)} (+- {np.std(fold_macrof1)})')
    print(f'> Loss: {np.mean(fold_losses)}')
    print('------------------------------------------------------------------------')