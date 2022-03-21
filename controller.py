from tkinter import Y
from model import Model
from view import View

class Controller():
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        if self.view.weak_Annotations:
            self.view.window.bind('<ButtonPress-1>', self.calc_circle_coords)
        else:
            self.view.window.bind('<ButtonPress-1>', self.on_press_draw_rect)
            self.view.window.bind('<ButtonRelease-1>', self.on_release_draw_rect)

        self.view.window.bind('d', self.model.delete_annotations)
        self.view.window.bind('n', self.model.next_img)
        self.view.window.bind('q', self.view.close_window)


    def calc_circle_coords(self,event):
        x,y = event.x, event.y
        r = 8
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        self.view.draw_circle(x0,y0,x1,y1)
        self.model.add_circle_coords(x,y)

    def on_press_draw_rect(self,event):
        self.x = event.x
        self.y = event.y
    
    def on_release_draw_rect(self,event):
        x0,y0 = self.x, self.y
        x1,y1 = event.x, event.y
        self.view.draw_rectangle(x0,y0,x1,y1)
        self.model.shape_IDs.append(self.view.ID)
        self.model.add_rect_coords(x0,y0,x1,y1)
     

    def start_ui(self):
        self.model.next_img()