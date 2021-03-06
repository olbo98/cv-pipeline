from email.mime import base
import random
import tkinter as tk
import os
from module import Module
from view import View
from controller import Controller
import numpy as np
from PIL import Image
from pool import Pool
import shutil #REMOVE


def main():
    path_to_labeled_imgs = "D:/Voi/cv-pipeline/labeled_imgs"
    path_to_labels = "D:/Voi/cv-pipeline/labels"
    path_to_unlabeled_imgs = "D:/Voi/cv-pipeline/unlabeled_imgs"
    path_to_weak_imgs = "D:/Voi/cv-pipeline/weak_imgs"
    path_to_testset = "D:/Voi/cv-pipeline/test_set"
    
    window = tk.Tk()
    view = View(window)
    module = Module(view, path_to_labeled_imgs, path_to_labels, path_to_unlabeled_imgs, path_to_weak_imgs, path_to_testset)

    view.start_UI(module.first_state)
    controller = Controller(module, view)
    window.mainloop()
    

if __name__ == "__main__":
    main()