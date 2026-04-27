import numpy as np

def is_xray(image):
    img = np.array(image)

    if len(img.shape) == 2:
        return True

    if len(img.shape) == 3:
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        diff = np.mean(abs(r - g)) + np.mean(abs(r - b)) + np.mean(abs(g - b))
        return diff < 15

    return False