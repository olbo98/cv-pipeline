from re import X
import tkinter as tk
from tracemalloc import start
from turtle import circle, width
from PIL import ImageTk, Image
from tkinter import ANCHOR, Button, Label
import os

class View():
    """
    A class that represents the programs interface. It shows the interface depending if the user annotates a strong or
    a weak annotation. The class also shows the image along with its annotations.
    """
    def __init__(self, window, app_width = 1920, app_height = 1080):
        self.window = window
        self.app_width = app_width
        self.app_height = app_height
        
        self.window.geometry(f'{self.app_width}x{self.app_height}')

        self.canvas_text = tk.Canvas(self.window, width = 400, height = 1080)
        self.canvas_text.place(x=0,y=0)
        self.canvas_image = tk.Canvas(self.window, width = 1280, height = 1080)
        self.canvas_image.place(x=400,y=0)

        self.canvas_main = tk.Canvas(self.window, width=self.app_width, height=self.app_height)
        self.canvas_training_sampling = tk.Canvas(self.window, width=self.app_width, height=self.app_height)

    
        self.id1 = self.canvas_text.create_text(180, 80)
        self.id2 = self.canvas_text.create_text(180, 110)
        self.id3 = self.canvas_text.create_text(180, 190)
        self.id4 = self.canvas_text.create_text(180, 150)
        
        self.window.attributes('-fullscreen', True)
        
   

    def draw_weak_Annotations(self):
        self.canvas_training_sampling.pack_forget()
        self.canvas_text.itemconfig(self.id1,text="Annotate by center-clicking an object", fill="black", font=('Helvetica 12 bold'))
        self.canvas_text.itemconfig(self.id2, text="Press 'n' to annotate the next image")
        self.canvas_text.itemconfig(self.id3, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))
        self.canvas_text.itemconfig(self.id4, text="Delete latest annotation by pressing 'd'", font=('Helvetica', 8))
    
        
    def draw_strong_Annotations(self):
        self.canvas_text.itemconfig(self.id1, text="Annotate by drawing a rectangle around an object", fill="black", font=('Helvetica 9 bold'))
        self.canvas_text.itemconfig(self.id2, text="Press 'n' to annotate the next image", font=(4))
        self.canvas_text.itemconfig(self.id3, text="Press 'q' if the annotation of the images are done", font=('Helvetica', 8))
        self.canvas_text.itemconfig(self.id4, text="Delete latest annotation by pressing 'd'", font=('Helvetica', 8))

    def start_UI(self,state1):
        self.canvas_main.pack()
        self.start_button = tk.Button(self.canvas_main, text="Start", padx=30, pady=15, command=state1)
        self.start_button.place(relx=0.5, rely=0.5, anchor="center")

    def start_training_sampling(self):
        self.canvas_main.pack_forget()
       
        self.canvas_training_sampling.pack()
        self.canvas_training_sampling.create_text(650 ,420 , text="TRAINING.....", font=('Helvetica 15 bold'))
        
        
        
        

    def draw_circle(self,x0,y0,x1,y1):
        self.ID = self.canvas_image.create_oval(x0,y0,x1,y1, width=3, fill='yellow')

    def draw_rectangle(self,x0,y0,x1,y1):
        self.ID = self.canvas_image.create_rectangle(x0,y0,x1,y1, width=3, outline='blue')
        
    def show_img(self,image):

        img = Image.open(image)
        img = img.resize((int(1280*0.899),int(966*0.899)))
        self.img = ImageTk.PhotoImage(img)
        
        self.canvas_image.create_image(0,0, anchor="nw",image=self.img)
        label = Label(image=self.img)
        label.image = self.img # keep a reference!
        
    def get_width_height_img(self):
        h = self.img.height()
        w = self.img.width()
        return h,w

    def close_window(self, event=None):
        self.window.destroy()
        
    