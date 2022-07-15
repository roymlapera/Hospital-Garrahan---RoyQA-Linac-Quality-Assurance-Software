import cv2
import matplotlib.pyplot as plt
import numpy as np

dpi = 72    #pixeles por pulgada (dots per inch)

colimador = cv2.imread('C:/Users/howar/Desktop/Garrahan/Dosimetria/starshot2020unique/2020 Unique/Giro Colimador.tif')

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

# Running the code

dpi = 

pts, final_image = get_points(colimador)
radio = np.sqrt( (pts[0][0]-pts[1][0])**2 + (pts[0][1]-pts[1][1])**2 ) * dpi
print (pts)
cv2.waitKey(0)