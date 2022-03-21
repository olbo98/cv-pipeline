
import tkinter as tk
import os
from model import Model
from view import View
from controller import Controller

def main():
    set_images = []
    path = "D:/Voi/cv-pipeline/cv-pipline/images"
    for image in os.listdir(path):
        set_images.append(image)
    
    window = tk.Tk()
    
    view = View(window, weak_Annotations=False)
    model = Model(view,set_images,path)
    controller = Controller(model,view)
    controller.start_ui()
    window.mainloop()

    
if __name__ == "__main__":
    main()