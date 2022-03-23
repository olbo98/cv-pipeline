from tkinter import Y
from model import Module
from view import View

class Controller():
    """
    A class that handles event when a user clicks on the interface
    """
    def __init__(self, model: Module, view: View):
        self.model = model
        self.view = view

        self.view.canvas_image.bind('<ButtonPress-1>', self.model.handle_buttonpress)
        self.view.canvas_image.bind('<ButtonRelease-1>', self.model.handle_buttonrelease)

        self.view.window.bind('d', self.model.delete_annotations)
        self.view.window.bind('n', self.model.next_img)
        self.view.window.bind('q', self.view.close_window)

