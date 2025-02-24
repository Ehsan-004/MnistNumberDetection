from sklearn import svm, metrics, preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
import dataset
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

        map(lambda metric: metric(self.y_test, predicted_labels), metrics)

    def getModel(self):
        return self.model
    
    def getSceler(self):
        return self.scaler
            


def main():
    dset = dataset.Dataset("data\\train.csv", "data")
    print(dset.getLen())
    data = dset.getData()

    print(data[0][:5])  

    print("len")
    print(len(data[0]))
    print(len(data[1]))
    print()

    mnistModel = MnistModel(data[0], data[1])
    # mnistModel = MnistModel(*data)
    
    dset_test = dataset.Dataset("data\\test.csv", "data")
    print(dset_test.getLen())
    data_test = dset_test.getData()

    print("now printssss")
    mnistModel.testModel(data_test[0], data_test[1], print_classification_report, print_confusion_matrix)



if __name__ == "__main__":
    main()
