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
    #labeled images
    path_to_labeled_imgs = "D:/Exjobb/cv-pipeline/labeled_images"
    path_to_labels = "D:/Exjobb/cv-pipeline/annotations"
    path_to_unlabeled_imgs = "D:/Exjobb/cv-pipeline/unlabeled_images"
    path_to_weak_imgs = "D:/Exjobb/cv-pipeline/weaklabeled_images"
    
    window = tk.Tk()
    view = View(window)
    module = Module(view, path_to_labeled_imgs, path_to_labels, path_to_unlabeled_imgs, path_to_weak_imgs)

    view.start_UI(module.first_state)
    controller = Controller(module, view)
    window.mainloop()
    

if __name__ == "__main__":
    main()