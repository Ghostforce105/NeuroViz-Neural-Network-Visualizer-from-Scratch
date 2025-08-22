import tkinter as tk
from tkinter import ttk
import sv_ttk
from create_nodes import Layer
import ast
from Forward_propogation import Forward_propogation as FPP
from Back_prop import BackwardPropagation as BP
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Toplevel
import numpy as np

N = 0
First_nodes = 0 
input_size = 0
output_size = 0
Layer_map = []
Confirm_set = 0
lr = 0.1
loss_array = []

def create():
    global layer_index, N, First_nodes, input_size, output_size, Layer_map,Confirm_set
    try:
        if(Confirm_set != 1):
            if N == 0:

                input_nodes = int(input_nodes_entry.get())
                input_nodes_entry.delete(0, tk.END)
                First_nodes = input_nodes
                
                input_size = input_nodes
                console_output.insert(tk.END, "Please enter another value to proceed to runtime.\n")
                console_output.insert(tk.END, "Please enter .. at end of input value if you want to run again and again one data\n")
                N += 1

            elif N == 1:

                input_nodes = int(input_nodes_entry.get())
                output_size = input_nodes
                input_nodes_entry.delete(0, tk.END)
                active = activation_options.get()
                if active == "None":
                    active = None
                else:
                    console_output.insert(tk.END, f"INSERTED {active} ACTIVATION FUNCTION\n")

                Layer1 = Layer(First_nodes, input_nodes, global_canvas,N,active)
                Layer1.First_layer()
                Layer_map.append(Layer1)  
                First_nodes = input_nodes
                N += 1

            else:
                input_nodes = int(input_nodes_entry.get())
                output_size = input_nodes
                input_nodes_entry.delete(0, tk.END)
                active = activation_options.get()
                if active == "None":
                    active = None
                else:
                    console_output.insert(tk.END, f"INSERTED {active} ACTIVATION FUNCTION\n")

                Layer2 = Layer(First_nodes, input_nodes, global_canvas,N,active)
                Layer2.Display_Nodes(input_nodes + 1, [], 1)
                Layer_map.append(Layer2)  
                First_nodes = input_nodes
                N += 1
        else:
            console_output.insert(tk.End,f"YOU CANT INPUT A LAYER NOW")

    except ValueError:
        console_output.insert(tk.END, "Please enter valid integers for nodes.\n")




def confirm():
    global input_data, output_data, epochs_entry, input_size, output_size,Confirm_set,input_data_list,output_data_list,lr,epoch_run,epochs,epoch_output,is_one
    lr = float(lr_entry.get())
    lr_entry.delete(0,tk.END)
    try:
        user_input = input_data.get("1.0", tk.END).strip()
        if not user_input:
            raise ValueError("Input data is empty.")

        is_one = False
        if user_input.startswith("[") and user_input.endswith("]"):
            input_data_list = ast.literal_eval(user_input)
        else:
            if user_input.endswith(".."): 
                is_one = True
                user_input = user_input[:-2].strip() 

            if user_input: 
                input_data_list = [float(x) for x in user_input.replace(",", " ").split()]
            else:
                input_data_list = []

        user_output = output_data.get("1.0", tk.END).strip()
        if not user_output:
            raise ValueError("Output data is empty.")
        if user_output.startswith("[") and user_output.endswith("]"):
            output_data_list = ast.literal_eval(user_output)
        else:
            output_data_list = [float(x) for x in user_output.replace(",", " ").split()]

        epochs = int(epochs_entry.get())

        if is_one:
            if len(input_data_list)/input_size < 1 or len(output_data_list)/output_size < 1:
                console_output.insert(tk.END, f"INVALID DATA LENGTH\n")
            else:
                console_output.insert(tk.END, f"Data successfully captured!\n")
                input_data.delete("1.0", tk.END)
                output_data.delete("1.0", tk.END)
                epochs_entry.delete(0, tk.END)
                Confirm_set = 1
                epoch_run = input_size
                epoch_output = output_size
        else:
            if ((input_size > len(input_data_list)/epochs and len(input_data_list) % input_size != 0) or
                (output_size > len(output_data_list)/epochs and len(output_data_list) % output_size != 0)):
                console_output.insert(tk.END, f"INVALID EPOCH COUNT EXCEEDING DATA LENGTH\n")
            else:
                console_output.insert(tk.END, f"Data successfully captured!\n")
                input_data.delete("1.0", tk.END)
                output_data.delete("1.0", tk.END)
                epochs_entry.delete(0, tk.END)
                Confirm_set = 1
                epoch_run = input_size
                epoch_output = output_size

    except ValueError as e:
        console_output.insert(tk.END, f"Error: {str(e)}\n")
    except Exception as e:
        console_output.insert(tk.END, f"Invalid input format: {str(e)}\n")


def run():
    global Confirm_set,output_data_list,lr,epoch_run,epochs,epoch_output,is_one,loss_array
    
    if(Confirm_set == 1):
        for i in range(epochs):
            Forpor = []
            if(is_one):
                Forpor = FPP(Layer_map,input_data_list,global_canvas,root)
            else:
                Forpor = FPP(Layer_map,input_data_list[(i*epoch_run):(i+1)*epoch_run],global_canvas,root)
            console_output.insert(tk.END,f"Forward Propogation was successfully Perfomed\n\n")
            console_output.insert(tk.END,f"Forward Propogation output was {Forpor.Final_value}\n\n")
            BACK_PROP = []
            if(is_one):
                BACK_PROP = BP(output_data_list,Forpor.Final_value,Layer_map,global_canvas,root,lr)
            else:
                BACK_PROP = BP(output_data_list[i*epoch_output:(i+1)*epoch_output],Forpor.Final_value,Layer_map,global_canvas,root,lr)
            console_output.insert(tk.END,f"Backward Propogation was successfully Perfomed\n\n")
            console_output.insert(tk.END,f"The Error was found to be {BACK_PROP.loss}\n\n")
            loss_array.append(BACK_PROP.loss)
    else:
        console_output.insert(tk.END,"ERROR YOU HAVE NOT CONFIRMED YOUR MODEL CANT PROCEED FURTURE\n")

def add_activation_layer():
    activation_function = activation_var.get()
    if activation_function:
        console_output.insert(tk.END, f"Added activation layer with {activation_function}.\n")
    else:
        console_output.insert(tk.END, "Please select an activation function.\n")

def show_graph():
    global loss_array, Confirm_set
    if Confirm_set == 1:
        plot_window = Toplevel(root)
        plot_window.title("Plot Window")
        plot_window.geometry("600x500")
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        x = np.linspace(1, len(loss_array), len(loss_array))
        ax.plot(x, loss_array, marker='o', color='blue')
        ax.set_title("LOSS ARRAY")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("LOSS VALUE")
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    else:
        console_output.insert(tk.END, "NOT CONFIRMED OR RUN YET\n")

root = tk.Tk()
root.state('zoomed')

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

global_canvas = tk.Canvas(main_frame, bg="black", highlightthickness=0)
global_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=global_canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

h_scrollbar = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=global_canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

global_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

scrollable_frame = tk.Frame(global_canvas, bg="black")
global_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

def on_canvas_configure(event):
    global_canvas.configure(scrollregion=global_canvas.bbox("all"))

scrollable_frame.bind("<Configure>", on_canvas_configure)


options_frame = tk.Frame(main_frame, width=400, bg="gray20")
options_frame.pack_propagate(False)
options_frame.pack(side=tk.RIGHT, fill=tk.Y)


tk.Label(options_frame, text="Layer Nodes (Input)", bg="gray20", fg="white", font=('Arial', 12, 'bold')).pack(pady=10)
input_nodes_entry = tk.Entry(options_frame)
input_nodes_entry.pack(pady=5)



tk.Label(options_frame, text="Add Activation Layer", bg="gray20", fg="white", font=('Arial', 12, 'bold')).pack(pady=10)
activation_var = tk.StringVar(value="None")
activation_options = ttk.Combobox(options_frame, textvariable=activation_var)
activation_options['values'] = ('ReLU', 'Tanh', 'Sigmoid','None')
activation_options.pack(pady=5)

tk.Button(options_frame, text="Add", command=create).pack(pady=5)





# Data I/O section
io_frame = tk.Frame(options_frame, bg="gray20")
io_frame.pack(pady=10, fill=tk.BOTH, expand=True)
notebook = ttk.Notebook(io_frame)
notebook.pack(fill=tk.BOTH, expand=True)
input_tab = tk.Frame(notebook, bg="gray20")
notebook.add(input_tab, text="Input Data")
input_data = tk.Text(input_tab, height=8, bg="black", fg="white", insertbackground='white')
input_data.pack(fill=tk.BOTH, expand=True)



# Output tab
output_tab = tk.Frame(notebook, bg="gray20")
notebook.add(output_tab, text="Output Data")
output_data = tk.Text(output_tab, height=8, bg="black", fg="white", insertbackground='white')
output_data.pack(fill=tk.BOTH, expand=True)


# Console tab
console_tab = tk.Frame(notebook, bg="gray20")
notebook.add(console_tab, text="Console")
console_output = tk.Text(console_tab, height=5, bg="black", fg="white")
console_output.pack(fill=tk.BOTH, expand=True)

#epoch button
tk.Label(options_frame, text="No. of Epochs", bg="gray20", fg="white", font=('Arial', 12, 'bold')).pack(pady=10)
epochs_entry = tk.Entry(options_frame)
epochs_entry.pack(pady=5)


#lr
tk.Label(options_frame, text="Learning rate", bg="gray20", fg="white", font=('Arial', 12, 'bold')).pack(pady=10)
lr_entry = tk.Entry(options_frame)
lr_entry.pack(pady=5)
run_frame = tk.Frame(options_frame, bg="gray20")
run_frame.pack(pady=5)
ttk.Button(run_frame, text="Confirm", style="Accent.TButton", command=confirm).pack(fill=tk.X)

# Run button
run_frame = tk.Frame(options_frame, bg="gray20")
run_frame.pack(pady=10)
ttk.Button(run_frame, text="RUN SIMULATION", style="Accent.TButton",command=run).pack(fill=tk.X)

run_frame = tk.Frame(options_frame, bg="gray20")
run_frame.pack(pady=10)
ttk.Button(run_frame, text="LOSS GRAPH", style="Accent.TButton",command=show_graph).pack(fill=tk.X)

sv_ttk.set_theme("dark")

layer_index = 0  
root.mainloop()
