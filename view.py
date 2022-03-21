import tkinter as tk
from turtle import circle
from PIL import ImageTk, Image
from tkinter import Label
import os

class View():
    def __init__(self, window, weak_Annotations, app_width = 1920, app_height = 1080):
        self.window = window
        self.weak_Annotations = weak_Annotations
        self.app_width = app_width
        self.app_height = app_height
        
        window.geometry(f'{self.app_width}x{self.app_height}')

        self.canvas = tk.Canvas(window, width=app_width, height=app_height)

        if weak_Annotations:
            self.draw_weak_Annotations()
        else:
            self.draw_strong_Annotations()

        
        self.canvas.pack()

        self.window.bind('q', self.close_window)
        self.window.attributes('-fullscreen', True)
        
    def draw_weak_Annotations(self):
        self.canvas.create_text(160, 80, text="Annotate by center-clicking an object", fill="black", font=('Helvetica 12 bold'))
        self.canvas.create_text(160, 110, text="Press 'n' to annotate the next image")
        self.canvas.create_text(160, 150, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))
        
    def draw_strong_Annotations(self):
        self.canvas.create_text(160, 80, text="Annotate by drawing a rectangle around an object", fill="black", font=('Helvetica 9 bold'))
        self.canvas.create_text(160, 110, text="Press 'n' to annotate the next image", font=(4))
        self.canvas.create_text(160, 150, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 7))

    def draw_circle(self,x0,y0,x1,y1):
        self.canvas.create_oval(x0,y0,x1,y1, width=3, fill='yellow')

    def draw_rectangle(self,x0,y0,x1,y1):
        self.canvas.create_rectangle(x0,y0,x1,y1, width=3, outline='blue')
        
    def show_img(self,image,path):

        img = ImageTk.PhotoImage(Image.open(os.path.join(path,image)))

        self.canvas.create_image(self.app_width/2, self.app_height/2, anchor="center", image=img)
        label = Label(image=img)
        label.image = img # keep a reference!
        label.pack()
        
    def close_window(self, event=None):
        self.window.destroy()

    def get_set_images(self):
        return 
        
    