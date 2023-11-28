import tkinter as tk
from tkinter import filedialog
import eel, os, random
# Set web files folder and optionally specify which file types to check for eel.expose()

eel.init('web')

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Open the file dialog
    return file_path

@eel.expose                         # Expose this function to Javascript
def handleinput(x):
    print(x)

@eel.expose
def get_file_from_user():
    file_path = select_file()
    return file_path



eel.say_hello_js('connected!') 

eel.start('index.html', size=(1400, 800))  