from tkinter import Y
from model import Model
from view import View

class Controller():
    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view

        self.view.window.bind('<ButtonPress-1>', self.model.handle_buttonpress)
        self.view.window.bind('<ButtonRelease-1>', self.model.handle_buttonrelease)

        self.view.window.bind('d', self.model.delete_annotations)
        self.view.window.bind('n', self.model.next_img)
        self.view.window.bind('q', self.view.close_window)

    def start_ui(self):
        self.model.next_img()