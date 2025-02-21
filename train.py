from sklearn import svm, metrics, preprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from pprint import pprint
import dataset
import time
import sys
import os



def main():
    dset = dataset.Dataset("data\\train.csv", "data")
    data = dset.getData()
    

    clf = svm.SVC(kernel="rbf")
    x = data[0]

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Preprocessing training data before training the model')

    x = preprocessing.StandardScaler().fit(x).transform(x.astype(float))

    t1 = time.time()

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Training the model')
    
    clf.fit(x, data[1])
    t2 = time.time()

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Model has been trained after {t2-t1:.2f} s')


    dset_test = dataset.Dataset("data\\test.csv", "data")
    data_test = dset_test.getData()

    x_test = data_test[0]
    x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))

    predict_label = clf.predict(x_test)

    fs = metrics.f1_score(data_test[1], predict_label, average="micro")
    print(f"f1 score: {fs}")
    clr = metrics.classification_report(data_test[1], predict_label)
    print(f"classification report: {clr}")
    cm = metrics.confusion_matrix(data_test[1], predict_label)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.show()
    
    # cm = metrics.confusion_matrix(data_test[1], predict_label)

    # disp = metrics.ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plt.show()



if __name__ == "__main__":
    main()
