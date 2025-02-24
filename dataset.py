from multiprocessing import Pool, Manager
from datetime import datetime
from glob import glob
from tqdm import tqdm
from PIL import Image   
import pandas as pd
import numpy as np
import time
import sys
import os
from pathlib import Path


def create_csv_file(path_to_main_dir, label, extension):
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] processing files with label: {label}')
    
    # image_paths = glob(os.path.join(path_to_main_dir, str(label), f"*.{extension}"))

    # list(map(lambda p: os.path.normpath(p), list(Path(path_to_main_dir).rglob(f"*.{extension}"))))

    image_paths = [os.path.normpath(p) for p in Path(path_to_main_dir, str(label)).rglob(f"*.{extension}")]

    
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {len(image_paths)} files are ready to process in {path_to_main_dir}')

    # d = dict()
    # for pat in image_paths:
    #     d[pat] = label
    
    return image_paths, [label] * len(image_paths)



def process_files(path, csv_file_name):
    t1 = time.time()

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] operations started. processing directory: {path}')

    # result = dict()

    args = [(path, i, "jpg") for i in range(10)]

    oip = list()
    oil = list()

    with Manager() as manager:
        image_pathes = manager.list()
        image_labels = manager.list()
        with Pool(os.cpu_count()) as p: 
            res = p.starmap(create_csv_file, args)

        for paths, labels in tqdm(res):
            image_pathes.extend(paths)
            image_labels.extend(labels)

        oip = list(image_pathes)
        oil = list(image_labels)

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] writing data to disk')

    pd.DataFrame({
        "path": oip,
        "label": oil
    }).to_csv(csv_file_name, index=False)
    t2 = time.time()

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] operation for {path} finished after {t2-t1:.2f} s')    
   


def read_path_labels(path):

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] processing file {path}')

    t1 = time.time()
    df = pd.read_csv(path)
    imgs_paths = list(df['path'])
    labels = list(df['label'].apply(str))

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] process completed after {time.time()-t1:.2f} s')

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
        image_path = os.path.normpath(os.path.join(self.dataDir, self.paths[index]))
        item = {
            "image": load_image(image_path),
            "label": self.labels[index]
        }
        return item


    def getData(self):  # (list of images, list of labels)
        images = [] 
        labels = []
        
        for i in tqdm(range(self.getLen())): #(self.getLen()):
            d = self.getImage(i)
            images.append(d["image"])
            labels.append(d["label"])
        
        return np.array(images), np.array(labels)
    
    def getValueCounts(self):

        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))



if __name__ == "__main__":  
    from pprint import pprint

    process_files("data/train", "data/train.csv")
    process_files("data/test", "data/test.csv")
    

    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] start')
    t1 = time.time()
    t = Dataset("data\\train.csv", "")
    f = t.getData()

    print(f[0][:5])

    pprint(t.getValueCounts())

    t2 = time.time()
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] done after {t2-t1:.2f} s')