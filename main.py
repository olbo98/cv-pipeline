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
    img_path = "D:/Voi/test_loop_imgs/20imgs"
    label_path = "D:/Voi/test_loop_imgs/labels20imgs"

    weak_labeled_pool = Pool([], [])
    labeled_pool = Pool([], [])
    unlabeled_pool = []
    unlabeled_path = "D:/Voi/cv-pipeline/cv-pipline/images"
    for image in os.listdir(unlabeled_path):
        unlabeled_pool.append(os.path.join(unlabeled_path,image))

    window = tk.Tk()
    view = View(window)
    module = Module(view, img_path, label_path, unlabeled_path, labeled_pool, weak_labeled_pool, unlabeled_pool)
    view.start_UI(module.first_state)
    controller = Controller(module, view)
    #module = Module(view, path)
    #controller = Controller(module,view)
    #view.start_UI()
    window.mainloop()
    rect_coords = module.get_rect_coords()
    print(rect_coords)
    

if __name__ == "__main__":
    main()