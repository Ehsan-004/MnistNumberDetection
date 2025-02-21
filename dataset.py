from datetime import datetime
from glob import glob
from PIL import Image   
import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path


def create_csv_file(path_to_main_dir, label, extension):
    
    image_paths = glob(os.path.join(path_to_main_dir, str(label), f"*.{extension}"))

    list(map(lambda p: os.path.normpath(p), list(Path(path_to_main_dir).rglob(extension))))
    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {len(image_paths)} files are ready to process in {path}')

    d = dict()
    for pat in image_paths:
        d[pat] = label
    
    return d


def process_files(path, file_name):
    t1 = time.time()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] operations started. processing directory: {path}')
    result = dict()
    for i in range(2):
        tt = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] processing files with label: {i}')
        result.update(create_csv_file(os.path.normpath(path), i))
        ttt = time.time()
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] processing files with label: {i} after {ttt-tt:.2f} s is done!')


    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] writing data to disk')
    pd.DataFrame({
        "path": result.keys(),
        "label": result.values()
    }).to_csv(file_name, index=False)
    t2 = time.time()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] operation for {path} finished after {t2-t1:.2f} s')


def read_path_labels(path):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] processing file {path}')
    t1 = time.time()
    df = pd.read_csv(path)
    imgs_paths = list(df['path'])
    labels = list(df['label'].apply(str))
    t2 = time.time()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] process completed after {t2-t1:.2f} s')
    return imgs_paths, labels


def load_image(path):
    image = Image.open(path)
    array_image = np.array(image).reshape(-1)
    return array_image


class Dataset:
    def __init__(self, csv_path, data_directory):
        self.paths, self.labels = read_path_labels(csv_path)
        self.dataDir = os.path.normpath(data_directory)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Dataset instance is created and data is loaded to ram ...')


    def getLen(self):
        return len(self.labels)


    def getImage(self, index):
        item = {
            "image": load_image(os.path.join(self.dataDir, self.paths[index])),
            "label": self.labels[index]
        }
        return item


    def getData(self):  # (list of images, list of labels)
        images = [] 
        labels = []
        
        for i in range(self.getLen()): #(self.getLen()):
            d = self.getImage(i)
            images.append(d["image"])
            labels.append(d["label"])
        
        return np.array(images), np.array(labels)



if __name__ == "__main__":  
    from pprint import pprint
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start')
    t1 = time.time()
    t = Dataset("data\\train.csv", "data")
    f = t.getData()
    print(f[0][:5])

    t2 = time.time()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] done after {t2-t1:.2f} s')