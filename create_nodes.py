import tkinter as tk
from tkinter import ttk
import numpy as np
import sympy as sp
import random
startX = 60
startY = 400
gap = 60
global_canvas = None
temp = []

class Layer:
    def __init__(self, Input_Nodes, Output_Nodes, global_draw, N, Activation_function=None):
        global startX, global_canvas
        self.inodes = Input_Nodes
        self.onodes = Output_Nodes
        self.asyncfunc = Activation_function
        self.ionesarr = []
        self.onodesarr = []
        

        self.weight = np.divide(np.random.uniform(-1, 1, size=(Output_Nodes,Input_Nodes)),1000)
        self.bias = np.divide(np.random.uniform(-1, 1, size=Output_Nodes),1000)

        self.weight_sym = sp.Matrix(Output_Nodes, Input_Nodes, 
                                    lambda i, j: sp.symbols(f'w{N}_{i+1}_{j+1}'))
        self.bias_sym = sp.Matrix(Output_Nodes, 1, 
                                  lambda i, j: sp.symbols(f'b{N}_{i+1}'))
        
        self.Lines = []
        self.grad_weight = []
        self.grad_bias = []
        self.activation = []
        global_canvas = global_draw
        global_canvas.configure(scrollregion=global_canvas.bbox("all"))

    def First_layer(self):
        self.Display_Nodes(self.inodes + 1, [], 0)
        self.Display_Nodes(self.onodes + 1, [], 1)

    def Display_Line(self):
        global temp, global_canvas
        lines_list = []
        
        for i in temp:  
            temp_array = []
            for j in self.onodesarr:
                color = ["Red","orange"]
                Node_line = global_canvas.create_line(
                    i[0] + 20, i[1], j[0] + 20, j[1], fill=color[random.randint(0,1)], width=1
                )
                temp_array.append(Node_line)
            lines_list.append(temp_array)
        
        self.Lines = np.array(lines_list, dtype=object)
        temp = self.onodesarr

    def Display_Nodes(self, Nodes, nodearr, N):
        global startX, startY, gap, global_canvas, temp

        Nodes_Up = int(Nodes / 2)

        for i in range(Nodes_Up):
            y = startY - (i * gap)
            node_id = global_canvas.create_oval(startX, y-20, startX+40, y+20, fill="lightblue")
            nodearr.append([startX, y])

        for i in range(Nodes - Nodes_Up):
            y = startY + (i * gap)
            node_id = global_canvas.create_oval(startX, y-20, startX+40, y+20, fill="lightblue")
            nodearr.append([startX, y])

        if N == 0:
            temp = nodearr
        if N == 1:
            self.onodesarr = nodearr
            self.Display_Line()
        
        startX += 150
