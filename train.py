from sklearn import svm, metrics, preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
from pathlib import Path
import dataset
import joblib
import time
import sys
import os


def print_confusion_matrix(true_labels, pred_labels):
    cm = metrics.confusion_matrix(true_labels, pred_labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.show()

def print_classification_report(true_labels, pred_labels):
    clr = metrics.classification_report(true_labels, pred_labels)
    print("\n=== Classification Report ===\n")
    print(clr)  



class MnistModel:
    def __init__(self, x_train, y_train, model=svm.SVC(kernel="rbf")):

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Preprocessing training data before training the model')

        self.scaler = preprocessing.StandardScaler()
        self.x_train = self.scaler.fit_transform(x_train.astype(float))
        self.y_train = y_train
        self.model = model

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Training the model')

        t = time.time()
        self.model.fit(self.x_train, y_train)

        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Model has been trained after {time.time()-t:.2f} s')

    def testModel(self, x_test, y_test, *metrics):
        self.x_test = self.scaler.transform(x_test.astype(float))
        self.y_test = y_test
        predicted_labels = self.model.predict(self.x_test)

        for metric in metrics:
            metric(self.y_test, predicted_labels)

    def getModel(self):
        return self.model
    
    def getSceler(self):
        return self.scaler

    def saveModel(self, path):
        new_path = str(Path(path).with_suffix(".pkl"))
        joblib.dump(self.model, new_path)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Model has been saved in : {new_path}')
    
    def saveScaler(self, path):
        new_path = str(Path(path).with_suffix(".pkl"))
        joblib.dump(self.scaler, new_path)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Scaler has been saved in : {new_path}')

            


def main():
    dset = dataset.Dataset("data\\train.csv", "")
    data = dset.getData()

    mnistModel = MnistModel(*data)

    mnistModel.saveModel("model.pkl")
    mnistModel.saveScaler("scaler.pkl")
    
    dset_test = dataset.Dataset("data\\test.csv", "")
    data_test = dset_test.getData()

    mnistModel.testModel(*data_test, print_classification_report, print_confusion_matrix)


if __name__ == "__main__":
    main()
