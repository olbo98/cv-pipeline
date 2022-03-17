import tkinter as tk
from turtle import circle
from PIL import ImageTk, Image
import sys
from tkinter import Label

class Interface():
    def __init__(self, window,  set_images, app_width = 1920, app_height = 1080):
        self.window = window
        self.app_width = app_width
        self.app_height = app_height
        window.geometry(f'{self.app_width}x{self.app_height}')

        self.canvas = tk.Canvas(window, width=app_width, height=app_height)
        self.canvas.pack()
        self.circle_coords = []
        self.circle_coords_img = []
        self.set_images = iter(set_images)
        self.canvas.create_text(160, 80, text="Annotate by center-clicking an object", fill="black", font=('Helvetica 12 bold'))
        self.canvas.create_text(160, 110, text="Press 'n' to annotate the next image")
        self.canvas.create_text(160, 150, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))
        self.window.bind('<ButtonPress-1>', self.draw_circle)
        self.window.bind('n', self.show_img)
        self.window.bind('q', self.close_window)
        self.canvas.pack()
        self.window.attributes('-fullscreen', True)
        
        

    def draw_circle(self,event):
        x,y = event.x, event.y
        r = 8
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        self.canvas.create_oval(x0,y0,x1,y1, width=3, fill='yellow')
        self.circle_coords_img.append([x,y])
        
    def show_img(self, event=None):
        try:
            self.circle_coords_img = []
            image = next(self.set_images)
            self.circle_coords.append(self.circle_coords_img)
        except StopIteration:
            return

        img = ImageTk.PhotoImage(Image.open(image))

        self.canvas.create_image(self.app_width/2, self.app_height/2, anchor="center", image=img)
        label = Label(image=img)
        label.image = img # keep a reference!
        label.pack()
        

        
        
        
    def get_circle_coords(self):
        return self.circle_coords

    def close_window(self, event=None):
        self.window.destroy()

        
    