import numpy as np
import random

class BackwardPropagation:
    def __init__(self, user_output, FP_output, Layer_map,global_canvas,root, lr=0.1):
        self.y = np.array(user_output)  
        self.yhat = np.array(FP_output)  
        self.Layer = Layer_map
        self.lr = lr
        self.global_canvas= global_canvas
        self.root = root
        
        self.loss = self.backwards()
    
    def activation_derivative(self, activation_type, activation_values, z_values=None):
        if activation_type == "Sigmoid":
            return activation_values * (1 - activation_values)
        elif activation_type == "Tanh":
            return 1 - activation_values**2
        elif activation_type == "ReLU":
            if z_values is not None:
                return (z_values > 0).astype(float)
            else:
                return (activation_values > 0).astype(float)
        else:  
            return np.ones_like(activation_values)
    
    def get_layer_input(self, layer_index):
        layer = self.Layer[layer_index]
        
        if layer_index == 0:
            input_candidates = [
                'input', 'input_data', 'x', 'inputs', 
                'input_activation', 'layer_input', 'raw_input'
            ]
            
            for attr_name in input_candidates:
                if hasattr(layer, attr_name):
                    input_val = getattr(layer, attr_name)
                    if input_val is not None:
                        return np.array(input_val)
            available_attrs = [attr for attr in dir(layer) if not attr.startswith('_')]
            raise ValueError(f"First layer input not found. Available attributes: {available_attrs}")
        
        else:

            prev_layer = self.Layer[layer_index - 1]
            activation_candidates = ['activation', 'output', 'a', 'result']
            
            for attr_name in activation_candidates:
                if hasattr(prev_layer, attr_name):
                    activation_val = getattr(prev_layer, attr_name)
                    if activation_val is not None:
                        return np.array(activation_val)
            
            raise ValueError(f"Previous layer activation not found for layer {layer_index}")
    
    def backwards(self):
        if isinstance(self.y, list):
            self.y = np.array(self.y)
        if isinstance(self.yhat, list):
            self.yhat = np.array(self.yhat)
        
        if self.y.ndim == 0:
            self.y = self.y.reshape(1)
        if self.yhat.ndim == 0:
            self.yhat = self.yhat.reshape(1)
            
        m = len(self.y)

        dL_da = (2.0 / m) * (self.yhat - self.y)
 
        for i in range(len(self.Layer)-1, -1, -1):
            layer = self.Layer[i]

            try:
                W = np.array(layer.weight)
                b = np.array(layer.bias)
                activation = np.array(layer.activation)
            except AttributeError as e:
                print(f"Layer {i} missing required attribute: {e}")
                continue
            
            activation_type = getattr(layer, 'asyncfunc', 'linear')
            
            z = getattr(layer, 'z', None)
            
 
            activation_deriv = self.activation_derivative(activation_type, activation, z)
    
            delta = dL_da * activation_deriv
            
            try:
                layer_input = self.get_layer_input(i)
            except ValueError as e:
                print(f"Error getting input for layer {i}: {e}")
                layer_input = activation
            
            if delta.ndim == 0:
                delta = delta.reshape(1)
            if layer_input.ndim == 0:
                layer_input = layer_input.reshape(1)
            
            if len(delta.shape) == 1 and len(layer_input.shape) == 1:
                dW = np.outer(delta, layer_input) / m
            else:
                dW = delta.reshape(-1, 1) @ layer_input.reshape(1, -1) / m
            
            db = delta / m
            
            try:
                layer.weight = W - self.lr * dW
                layer.bias = b - self.lr * db
            except ValueError as e:
                print(f"Shape mismatch in layer {i}: {e}")
                print(f"W shape: {W.shape}, dW shape: {dW.shape}")
                print(f"b shape: {b.shape}, db shape: {db.shape}")
            if i > 0:
                dL_da = W.T @ delta
            for line_group in self.Layer[i].Lines:
                for line_id in line_group:
                    self.global_canvas.itemconfig(line_id, fill="yellow")
                    self.root.update()
                    color = ["Red","orange"]
                    self.root.after(2000, lambda l=line_id: self.global_canvas.itemconfig(l, fill=color[random.randint(0,1)]))
        loss = np.mean((self.yhat - self.y)**2)
        return loss

