import tkinter as tk
from tkinter import ttk
import sv_ttk
import numpy as np
import random

class Forward_propogation:
    def __init__(self, Layer_map, user_input, global_canvas, root):
        self.Layer_map = Layer_map
        self.user_input = np.array(user_input, dtype=float) 
        self.Final_value = []
        self.ans = self.FP_calc(global_canvas, root)

    def FP_calc(self, global_canvas, root):
        z = 0
        for layer in self.Layer_map:

            z = np.dot(layer.weight, self.user_input) + layer.bias


            if layer.asyncfunc == "ReLU":
                z = np.maximum(0, z)


            elif layer.asyncfunc == "Sigmoid":
                z = 1 / (1 + np.exp(-z))
 

            elif layer.asyncfunc == "Tanh":
                z = np.tanh(z)


  
            self.user_input = z
            for line_group in layer.Lines:
                for line_id in line_group:
                    global_canvas.itemconfig(line_id, fill="blue")
                    root.update()
                    color = ["Red","green"]
                    root.after(2000, lambda l=line_id: global_canvas.itemconfig(l, fill=color[random.randint(0,1)]))
            layer.activation = z
        self.Final_value = z
        return self.Final_value
