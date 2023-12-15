import tkinter as tk
from tkinter import filedialog
import eel, os, random
import matplotlib.pyplot as plt
import pandas as pd
import base64
from io import BytesIO
import mlModule as mlM

# Set web files folder and optionally specify which file types to check for eel.expose()



eel.init('web')

def generate_graph_base64(fig, width=600, height=400):
    """

    Encodes a matplotlib figure object into a base64 string. 
    :param fig: matplotlib figure object
    :param width: Width of the image in pixels, default is 600
    :param height: Height of the image in pixels, default is 400
    :return: str, base64 encoded string of the graph image
    
    """
    try:
        dpi = 70  # Set the dots per inch to render
        fig_width = width / dpi  # Convert width from pixels to inches
        fig_height = height / dpi  # Convert height from pixels to inches

        # Update the size of the given figure
        fig.set_size_inches(fig_width, fig_height)

        # Convert plot to PNG image
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)

        # Encode the image in base64 and return it
        image_png = buffer.getvalue()
        graph_base64 = "data:image/png;base64," + base64.b64encode(image_png).decode('utf-8')
        buffer.close()

        return graph_base64
    except Exception as e:
        return f"An error occurred: {e}"

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    return file_path


@eel.expose                         # Expose this function to Javascript
def handleinput(x):
    print(x)

@eel.expose
def get_file_from_user(width=600, height=400):
    file_path = select_file()
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    
    sales_fig, inflation_fig, unemployment_fig, temperature_fig = mlM.generate_graphs(file_path)
    base64_sales = generate_graph_base64(sales_fig)
    base64_inflation = generate_graph_base64(inflation_fig)
    base64_unemployment = generate_graph_base64(unemployment_fig)
    base64_temperature = generate_graph_base64(temperature_fig)
    
    return base64_sales, base64_inflation, base64_unemployment, base64_temperature, file_path

@eel.expose
def predict_use_linear_regressor(file_path):
    print(file_path)
    sales_fig, accuracy_fig = mlM.use_linear_regressor(file_path)
    base64_sales = generate_graph_base64(sales_fig)
    base64_accuracy = generate_graph_base64(accuracy_fig)
    return base64_sales, base64_accuracy

@eel.expose
def predict_use_decision_tree_regressor(file_path):
    print(file_path)
    sales_fig, accuracy_fig = mlM.use_decision_tree_regressor(file_path)
    base64_sales = generate_graph_base64(sales_fig)
    base64_accuracy = generate_graph_base64(accuracy_fig)
    return base64_sales, base64_accuracy

@eel.expose
def predict_use_random_forest_regressor(file_path):
    print(file_path)
    sales_fig, accuracy_fig = mlM.use_random_forest_regressor(file_path)
    base64_sales = generate_graph_base64(sales_fig)
    base64_accuracy = generate_graph_base64(accuracy_fig)
    return base64_sales, base64_accuracy



eel.say_hello_js('connected!') 

eel.start('index.html', size=(1800, 1000))  