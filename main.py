import tkinter as tk
import os
from module import Module
from view import View
from controller import Controller
import numpy as np
from PIL import Image

def main():
    set_images = []
    
    path = "D:/Voi/test_loop_imgs/20imgs"
    for image in os.listdir(path):
        set_images.append(image)
    

    
    window = tk.Tk()
    view = View(window)
    module = Module(view, path)
    module.test_train()
    #module.prepare_imgs(set_images)
    controller = Controller(module,view)

if __name__ == "__main__":
    main()