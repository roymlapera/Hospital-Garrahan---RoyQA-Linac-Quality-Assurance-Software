from tkinter import *
from tkinter.ttk import Combobox
from tkinter import ttk
from tkinter import filedialog, messagebox

def ElegirArchivos():
        Tk().withdraw()
        filename =  filedialog.askopenfilename(title='Seleccionar imágen',
                                                filetypes=(('Imágen TIF','*.tif'), ('Imágen TIFF','*.tiff'), ('Todos los archivos','*.')))
        return filename

# tk.Tk().withdraw()
#     filename =  filedialog.askopenfilename(title='Seleccionar imagen', 
#                                                 filetypes=(('Imagen TIF','*.tif'), ('Imagen TIFF','*.tiff'), ('All Files','*.')))
#     return filename

# def select_file():
#     filetypes = (('Archivo TIF', '*.tif'),('Archivo TIFF', '*.tiff'),('Todos los archivos', '*.*'))

#     filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)

#     showinfo(title='Selected File',message=filename)

filename = ElegirArchivos()


window=Tk()
var = StringVar()
var.set("one")
data=("Colimador", "Camilla", "Gantry")
# cb=Combobox(window, values=data)
# cb.place(x=60, y=150)

lb=Listbox(window, height=5, selectmode='multiple')
for num in data:
    lb.insert(END,num)
lb.place(x=250, y=150)

v0=IntVar()
v0.set(1)
r1=Radiobutton(window, text="Colimador", variable=v0,value=1)
r2=Radiobutton(window, text="Camilla", variable=v0,value=2)
r3=Radiobutton(window, text="Gantry", variable=v0,value=3)
r1.place(x=100, y=50)
r2.place(x=180, y=50)
r3.place(x=260, y=50)
                

open_button = ttk.Button(window,text='Open a File',command=select_file)
open_button.pack(expand=True)
open_button.place(x=10, y=10)

window.title('Análisis Starshot')
window.geometry("1200x750+10+10")
window.mainloop()

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo

# create the root window
root = tk.Tk()
root.title('Tkinter Open File Dialog')
root.resizable(False, False)
root.geometry('300x150')



