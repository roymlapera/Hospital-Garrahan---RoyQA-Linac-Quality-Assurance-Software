from tkinter import filedialog, messagebox
from pylinac import Starshot
import tkinter as tk
import cv2

#NO FUNCIONA PORQUE APARECE UN ERROR CON LOS ARCHIVOS DE IMAGEN TIFF. ABRIL 2022

def TestBox():
    messagebox.showinfo(message="Todo ok!",title="Mensaje")

class Starshot_class:

    def ElegirArchivos(self):
        tk.Tk().withdraw()
        filename =  filedialog.askopenfilename(title='Seleccionar imagen', 
                                                filetypes=(('Imagen TIF','*.tif'), ('Imagen TIFF','*.tiff'), ('All Files','*.')))
        return filename

    def __init__(self, tipo, dpi):        
        self.tipo = tipo                           
        self.path = self.ElegirArchivos()
        # self.starshot = Starshot(self.path)
        self.dpi = dpi

dpi = 72

colimador = Starshot_class('colimador', dpi)

print(colimador.path)

myystar = Starshot(colimador.path)

# Displaying the image
# cv2.imshow('Starshot', ss_colimador.imagen)
# cv2.waitKey(0)


colimador.starshot.analyze()
print(colimador.starshot.results())
colimador.starshot.plot_analyzed_image()





