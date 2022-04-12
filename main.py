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

    window = tk.Tk()
    view = View(window)
    module = Module(view, labeled_img_path, labeled_label_path, unlabeled_path)
    view.start_UI(module.first_state)
    controller = Controller(module, view)
    window.mainloop()
    

if __name__ == "__main__":
    main()