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
    module = Module(view, path, set_images)
    module.prepare(set_images)
    view.start_UI(module.first_state)
    controller = Controller(module, view)
    #module = Module(view, path)
    #controller = Controller(module,view)
    #view.start_UI()
    window.mainloop()
    

if __name__ == "__main__":
    main()