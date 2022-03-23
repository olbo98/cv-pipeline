import tkinter as tk
import os
from model import Module
from view import View
from controller import Controller

def prepare_imgs(set_images):
    circle_coords = {}
    rect_coords = {}
    set_images = iter(set_images)
    while True:
        try:
            image = next(set_images)
            circle_coords[image] = []
            rect_coords[image] = []
        except StopIteration:
            break
    return circle_coords, rect_coords

def main():
    set_images = []
    
    path = "D:/Voi/cv-pipeline/cv-pipline/images"
    for image in os.listdir(path):
        set_images.append(image)
    circle_coords, rect_coords = prepare_imgs(set_images)

    
    window = tk.Tk()
    view = View(window)
    model = Module(view, path, circle_coords, rect_coords)
    controller = Controller(model,view)

if __name__ == "__main__":
    main()