import tkinter as tk
import os
from module import Module
from view import View
from controller import Controller
import numpy as np
from PIL import Image
from pool import Pool


def main():
    #labeled images
    labeled_img_path = "D:/Exjobb/cv-pipeline/labeled_images"
    labeled_label_path = "D:/Exjobb/cv-pipeline/annotations"
    unlabeled_path = "D:/Exjobb/cv-pipeline/unlabeled_images"
    weak_labeled_pool = Pool([], [])
    labeled_pool = Pool([], [])
    unlabeled_pool = []
    
    
    for image in os.listdir(unlabeled_path):
        unlabeled_pool.append(os.path.join(unlabeled_path,image))
    for (img, label_file) in zip(os.listdir(labeled_img_path),os.listdir(labeled_label_path)):
        img_labels = []
        with open(labeled_label_path + "/" + label_file, "r") as f:
            for line in f:
                label = line.split(" ")
                label = [float(x) for x in label]
                img_labels.append(label)
        labeled_pool.add_sample(img, img_labels)

    window = tk.Tk()
    view = View(window)
    module = Module(view, labeled_img_path, labeled_label_path, unlabeled_path, labeled_pool, weak_labeled_pool, unlabeled_pool)
    view.start_UI(module.first_state)
    controller = Controller(module, view)
    window.mainloop()
    

if __name__ == "__main__":
    main()