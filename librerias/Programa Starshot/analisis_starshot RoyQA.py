import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize
from itertools import combinations
from tkinter import filedialog, messagebox
import tkinter as tk

def TestBox(): messagebox.showinfo(message="Todo ok!",title="Mensaje")

class Starshot:

    def ElegirArchivos(self):
        tk.Tk().withdraw()
        filename =  filedialog.askopenfilename(title='Seleccionar imagen', 
                                                filetypes=(('Imagen TIF','*.tif'), ('Imagen TIFF','*.tiff'), ('All Files','*.')))
        return filename

    def __init__(self, tipo, dpi):        
        self.tipo = tipo                           
        self.path = self.ElegirArchivos()
        self.imagen = cv2.imread(self.path)
        self.dpi = dpi


dpi = 72    #pixeles por pulgada (dots per inch)
radio = 180 #mm
threshold = 100
ancho = 5 #mm

starshot_tipo = 'colimador'
starshot_tipo = 'camilla'
starshot_tipo = 'gantry'

colimador = Starshot(tipo=starshot_tipo, dpi=dpi)

imagen = colimador.imagen

btn_down = False

def get_points(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['lines'] = []

    # Set the callback function for any mouse event
    cv2.imshow("Image", im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    # Convert array to np.array in shape n,2,2
    points = np.uint16(data['lines'])

    return points, data['im']

def mouse_handler(event, x, y, flags, data):
    global btn_down

    if event == cv2.EVENT_LBUTTONUP and btn_down:
        #if you release the button, finish the line
        btn_down = False
        data['lines'][0].append((x, y)) #append the second point

        radio = int(  np.sqrt( (data['lines'][0][0][0]-data['lines'][0][1][0])**2 + 
              (data['lines'][0][0][1]-data['lines'][0][1][1])**2 )   )

        xi = int(data['lines'][0][0][0])
        yi = int(data['lines'][0][0][1])
        cv2.circle(data['im'], (xi, yi), 2, (0, 0, 255), thickness=1)
        cv2.circle(data['im'],(xi,yi),radio,(0,0,255), thickness=1)

        cv2.imshow("Image", data['im'])

    elif event == cv2.EVENT_MOUSEMOVE and btn_down:
        #thi is just for a line visualization
        image = data['im'].copy()
        cv2.line(image, data['lines'][0][0], (x, y), (0,0,0), 1)
        cv2.imshow("Image", image)

    elif event == cv2.EVENT_LBUTTONDOWN and len(data['lines']) < 1:
        btn_down = True
        data['lines'].insert(0,[(x, y)]) #prepend the point
        cv2.imshow("Image", data['im'])

pts, final_image = get_points(imagen)
# cv2.waitKey(0)

xi = float(pts[0][0][0])*25.4/dpi
yi = float(pts[0][0][1])*25.4/dpi
xf = float(pts[0][1][0])*25.4/dpi
yf = float(pts[0][1][1])*25.4/dpi

radio = np.sqrt( (xi-xf)**2 + (yi-yf)**2 )
print(radio)

def Procesamiento(imagen, radio=radio, ancho=ancho, dpi=dpi):
  #Invierte y seleccionar ROI

  imagen_invertida = np.invert(imagen[:,:,1])   # Green 

  radio_px = int(radio*dpi/25.4)
  ancho_px = int(ancho*dpi/25.4)

  # roi_size = radio_px
  size_r = imagen_invertida.shape[0]  #nro filas
  size_c = imagen_invertida.shape[1]  #nro columnas
  roi_size = size_r #asumo que la imagen siempre tiene mas columnas que filas

  roi = imagen_invertida[ int((size_r-roi_size)/2):int((size_r+roi_size)/2),
					                int((size_c-roi_size)/2):int((size_c+roi_size)/2) ]

  # roi = imagen_invertida

  # plt.imshow(imagen_invertida)
  # plt.show()
  # plt.imshow(roi)
  # plt.show()

  return roi.astype(np.float32)

img = Procesamiento(imagen)

def PerfilStarshots(img, centro_x, centro_y, radio=radio, ancho=ancho, dpi=dpi):
  #--- the following holds the square root of the sum of squares of the image dimensions ---
  #--- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---

  centro_x = int(img.shape[0]/2)
  centro_y = int(img.shape[1]/2)

  radio_max_polar = int((radio+ancho/2)*dpi/25.4)

  ancho_px = int(ancho*dpi/25.4)

  polar_image = cv2.linearPolar(img,(centro_x, centro_y), radio_max_polar, cv2.WARP_FILL_OUTLIERS)

  polar_image = polar_image.astype(np.uint8)

  # plt.imshow(polar_image)
  # cv2.imshow("Polar Image", polar_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  perfil = polar_image[:,-ancho_px:]

  perfil = np.mean(perfil, axis=1)

  phi = np.arange(0,perfil.shape[0])*360/perfil.shape[0] #angulo en grados pero cada 360/len(perfil)

  plt.plot(phi,perfil)
  plt.ylabel('Intensidad de pixel')
  plt.ylim([perfil.min(), perfil.max()])
  plt.xlim([phi.min(),phi.max()])
  plt.xlabel('Angulo polar [º]')
  plt.show()

  return phi, perfil

phi, perfil = PerfilStarshots(img, xi, yi, radio=radio, ancho=ancho, dpi=dpi)
 
def Coordenadas(phi,perfil,threshold):

  def _gauss(x, a, mu, sig):
      return a*np.exp(-(x-mu)**2/(2*sig**2))

  def gauss_fit(x, y, xmin, xmax):
      fitx = x[(x>xmin)*(x<xmax)]
      fity = y[(x>xmin)*(x<xmax)]
      mu = np.sum(fitx*fity)/np.sum(fity)
      sig = np.sqrt(np.sum(fity*(fitx-mu)**2)/np.sum(fity))

      popt, pcov = curve_fit(_gauss, fitx, fity, p0=[max(fity), mu, sig])
      return popt[0], popt[1], popt[2]

  def CondicionaPerfil(phi,perfil,threshold):
    if perfil[0]>promedio:  #este es el caso en el que el primer pico esta cortado
      for i in range(len(perfil)):
        if (perfil[i]<promedio): 
          n = i+1
          break
      perfil2 = np.concatenate( ( perfil[n:], perfil[:n]) , axis=0 )
      aux     = phi[:n] + 360.0
      phi2    = np.concatenate( ( phi[n:], aux) , axis=0 )
    else:
      perfil2 = perfil
      phi2    = phi
    return phi2,perfil2

  promedio = threshold
  idx = []
  parametros = []
  rangos = []

  phi2,perfil2 = CondicionaPerfil(phi,perfil,promedio)

  for i in range(len(perfil2)):
    if(perfil2[i]>=promedio):
      idx.append(i)
    if(perfil2[i]<promedio and idx):
      pmin = phi2[ idx[0]  ]
      pmax = phi2[ idx[-1] ]
      rangos.append( (pmin,pmax) )
      parametros.append( gauss_fit(phi2[idx], perfil2[idx], pmin, pmax) )
      idx = []
      continue

  #Calculo angulos de los spokes: theta
  theta = np.array(parametros).transpose()[1]
  for i in range(theta.shape[0]):
    if theta[i]>=360: theta[i] = theta[i] - 360
  theta = np.sort(theta, axis=0)

  #calculo sus radios = r
  radius = [radio] * theta.shape[0]    #radio sale de las celdas anteriores #mm

  plt.plot(phi,perfil, linewidth=1)
  for i in range(len(parametros)):
    plt.axvline(x = theta[i] , color='g', linewidth=0.5)
  plt.ylabel('Intensidad de pixel')
  plt.ylim([perfil.min(), perfil.max()])
  plt.xlim([phi.min(),phi.max()])
  plt.xlabel('Angulo polar [º]')
  plt.show()

  return theta, radius

theta, radius = Coordenadas(phi,perfil,threshold)

#Calculo de las rectas

def CoeficientesRectas(theta,radius):

  rectas = []

  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  marker = ['.','o','*','D','+','v']
  colour = ['b','g','r','c','k','m']

  for i in range(theta.shape[0]//2):
    p1_r,p1_phi = ( radius[i]                   , theta[i]                   )
    p2_r,p2_phi = ( radius[i+theta.shape[0]//2] , theta[i+theta.shape[0]//2] )

    p1_x,p1_y = ( p1_r*np.cos(p1_phi*np.pi/180.0) , p1_r*np.sin(p1_phi*np.pi/180.0) )
    p2_x,p2_y = ( p2_r*np.cos(p2_phi*np.pi/180.0) , p2_r*np.sin(p2_phi*np.pi/180.0) )

    # plt.scatter(p1_x,p1_y, marker=marker[i], color=colour[i])
    # plt.scatter(p2_x,p2_y, marker=marker[i], color=colour[i])

    a = (p1_y - p2_y) / (p1_x - p2_x)
    b = p1_y - a * p1_x
    rectas.append( [a,b] )

  # x = np.arange(-100,100,2)

  # for i in range(len(rectas)):
  #   y = rectas[i][0] * x + rectas[i][1]
  #   if (i==0 or i==2 or i==3):  plt.plot(x,y)
  #   #plt.plot(x,y)

  # ax.set_aspect('equal', adjustable='box')
  # plt.ylim(1, 5) #mm
  # plt.xlim(1, 5) #mm
  # plt.grid('on')
  # plt.ylabel('Distancia [mm]')
  # plt.xlabel('Distancia [mm]')
  # plt.show()

  a = np.array(rectas)[:,0]
  b = np.array(rectas)[:,1]

  return a,b

a, b = CoeficientesRectas(theta,radius)

def SolucionStarshot(a,b, radio=radio):

  def d2(x): return (x[2]-x[0])**2+(x[3]-x[1])**2

  def EncuentraMenorCirculo(i,j,k,a,b):

    cons = ({'type': 'eq', 'fun': lambda x: a[i]*x[2] + b[i] - x[3] },
            {'type': 'eq', 'fun': lambda x: a[j]*x[4] + b[j] - x[5] },
            {'type': 'eq', 'fun': lambda x: a[k]*x[6] + b[k] - x[7] },
            {'type': 'eq', 'fun': lambda x: (x[4]-x[0])**2 + (x[5]-x[1])**2 - (x[6]-x[0])**2 - (x[7]-x[1])**2 },
            {'type': 'eq', 'fun': lambda x: (x[2]-x[0])**2 + (x[3]-x[1])**2 - (x[6]-x[0])**2 - (x[7]-x[1])**2 }
             )

    xcero = np.random.uniform(-50,50,8)
    bnds = ( (-radio,radio), (-radio,radio), (-radio,radio), (-radio,radio), 
             (-radio,radio), (-radio,radio), (-radio,radio), (-radio,radio) )

    res = minimize(d2,xcero,method='SLSQP', tol=1E-6, bounds=bnds, constraints=cons)

    return np.sqrt( (res.x[2]-res.x[0])**2 + (res.x[3]-res.x[1])**2 ), res.x[0], res.x[1]

  resultados = []

  for i,j,k in combinations(np.arange(a.shape[0]),3):
    resultados.append( EncuentraMenorCirculo(i,j,k,a,b) )

  resultados = np.array(resultados).T.tolist()

  idx_max = resultados[0].index(max(resultados[0]))

  diametro = resultados[0][idx_max]*2
  x0       = resultados[1][idx_max]
  y0       = resultados[2][idx_max]

  return diametro, x0, y0

diametro, x0, y0 = SolucionStarshot(a,b, radio=radio)

def Grafico(a,b,diametro,x0,y0):

  x = np.arange(-100,100,2)

  fig = plt.figure()
  ax = fig.add_subplot(111)

  for i in range(len(a)):
    y = a[i] * x + b[i]
    plt.plot(x,y)
  plt.plot(x0,y0, marker='o')
  a_Circle = plt.Circle( (x0,y0), diametro/2, fill=False)
  ax.add_artist(a_Circle)

  ax.set_aspect('equal', adjustable='box')
  plt.xlim(x0-2.5*diametro, x0+2.5*diametro) #mm
  plt.ylim(y0-2.5*diametro, y0+2.5*diametro) #mm
  plt.grid('on')
  plt.ylabel('Distancia [mm]')
  plt.xlabel('Distancia [mm]')
  plt.text(x0-2*diametro,y0-2*diametro, 'Diametro = '+str(round(diametro,1))+' mm', 
                                  fontsize = 12)
  plt.show()

Grafico(a,b,diametro,x0,y0)