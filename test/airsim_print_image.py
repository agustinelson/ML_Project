import airsim
import numpy as np
import os
import matplotlib.pyplot as plt

# Conéctate al cliente AirSim
client = airsim.MultirotorClient()

# Habilita las API de control
client.enableApiControl(True)

# Despega el dron
client.armDisarm(True)
client.takeoffAsync().join()

# Mueve el dron a una posición específica
client.moveToPositionAsync(0, 30, 0, 40).join()


responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
response = responses[0]

# Convierte la imagen a un array de numpy
img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)

# Cambia la forma del array para que sea una imagen de 3 canales H x W x 3
img_rgb = img1d.reshape(response.height, response.width, 3)

# La imagen original está volteada verticalmente
img_rgb = np.flipud(img_rgb)

# Guarda la imagen en un archivo usando matplotlib
plt.imsave('imagen_airsim.png', img_rgb)

# Aterriza el dron
client.landAsync().join()
client.reset()
    # Deshabilita las API de control
client.armDisarm(False)
client.enableApiControl(False)