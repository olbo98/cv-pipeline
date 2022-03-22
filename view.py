from re import X
import tkinter as tk
from turtle import circle
from PIL import ImageTk, Image
from tkinter import ANCHOR, Button, Label
import os

class View():
    def __init__(self, window, app_width = 1920, app_height = 1080):
        self.window = window
        self.app_width = app_width
        self.app_height = app_height
        self.draw_weak_labels = False
        
        self.window.geometry(f'{self.app_width}x{self.app_height}')

        self.canvas_image = tk.Canvas(self.window, width = 1520, height = 1080)
        self.canvas_image.place(x=400,y=0)
        self.canvas_text = tk.Canvas(self.window, width = 400, height = 1080)
        self.canvas_text.place(anchor=tk.NW,x=0,y=0)
        

        #if weak_Annotations:
         #   self.draw_weak_Annotations()
        #else:
         #   self.draw_strong_Annotations()

        self.canvas_text.create_text(180, 150, text="Delete latest annotation by pressing 'd'", font=('Helvetica', 8))
        
        self.window.attributes('-fullscreen', True)
        
    def draw_weak_Annotations(self):
        self.draw_weak_labels = True
        self.canvas_text.create_text(180, 80, text="Annotate by center-clicking an object", fill="black", font=('Helvetica 12 bold'))
        self.canvas_text.create_text(180, 110, text="Press 'n' to annotate the next image")
        self.canvas_text.create_text(180, 190, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))
        
    def draw_strong_Annotations(self):
        self.draw_weak_labels = False
        self.canvas_text.create_text(180, 80, text="Annotate by drawing a rectangle around an object", fill="black", font=('Helvetica 9 bold'))
        self.canvas_text.create_text(180, 110, text="Press 'n' to annotate the next image", font=(4))
        self.canvas_text.create_text(180, 190, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))

    def draw_circle(self,x0,y0,x1,y1):
        self.ID = self.canvas_image.create_oval(x0,y0,x1,y1, width=3, fill='yellow')

    def draw_rectangle(self,x0,y0,x1,y1):
        self.ID = self.canvas_image.create_rectangle(x0,y0,x1,y1, width=3, outline='blue')
        
    def show_img(self,image,path):

        img = ImageTk.PhotoImage(Image.open(os.path.join(path,image)))
        
        self.canvas_image.create_image(0,0, anchor="nw",image=img)
        label = Label(image=img)
        label.image = img # keep a reference!
        
        
    def close_window(self, event=None):
        self.window.destroy()
        
    