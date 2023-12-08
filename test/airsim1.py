import airsim
import cv2
import numpy as np
import time

# Conéctate al cliente AirSim
client = airsim.MultirotorClient()

# Habilita las API de control
client.enableApiControl(True)

# Despega el dron
client.armDisarm(True)
client.takeoffAsync().join()

# Mueve el dron a una posición específica
client.moveToPositionAsync(0, 30, 0, 40).join()

# Configura la ventana de visualización
cv2.namedWindow("Dron View", cv2.WINDOW_NORMAL)

try:
    while True:
        # Obtiene la imagen del dron
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        # Muestra la imagen en la ventana
        cv2.imshow("Dron View", img_rgb)

        # Verifica si ha llegado al destino
        if client.getMultirotorState().position.x_val > 9.5 and client.getMultirotorState().position.y_val < -9.5:
            # Reinicia la posición del dron
            client.moveToPositionAsync(0, 0, 0, 5).join()

        # Espera un breve periodo de tiempo para reducir la tasa de refresco
        time.sleep(10)

except KeyboardInterrupt:
    pass

finally:
    # Aterriza el dron
    client.landAsync().join()
    client.reset()
    # Deshabilita las API de control
    client.armDisarm(False)
    client.enableApiControl(False)

    # Cierra la ventana de visualización
    cv2.destroyAllWindows()
