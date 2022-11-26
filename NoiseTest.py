import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage import io           # Only needed for web grabbing images, use cv2.imread for local images


def is_valid(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    ##### Just for visualization and debug; remove in final
    plt.plot(s)
    plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
    plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
    plt.show()
    ##### Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.5
    return s_perc > s_thr


# Read example images; convert to grayscale
# noise1 = cv2.cvtColor(io.imread('https://i.stack.imgur.com/Xz9l0.png'), cv2.COLOR_RGB2BGR)
# noise2 = cv2.cvtColor(io.imread('https://i.stack.imgur.com/9ZPAj.jpg'), cv2.COLOR_RGB2BGR)
# valid = cv2.cvtColor(io.imread('https://i.stack.imgur.com/0FNPQ.jpg'), cv2.COLOR_RGB2BGR)

# for img in [noise1, noise2, valid]:
#     print(is_valid(img))


# Read example images; convert to grayscale
# noise1 = cv2.cvtColor(io.imread('https://i.stack.imgur.com/Xz9l0.png'), cv2.COLOR_RGB2BGR)
noise2 = cv2.cvtColor(cv2.imread("C:\\Users\\abisht\\Downloads\\RealtimeDenoising\\WebRTCCapturesDataset\\GT\\im2.png"), cv2.COLOR_RGB2BGR)
# noise3 = cv2.cvtColor(io.imread('C:\Users\abisht\Downloads\RealtimeDenoising\WebRTCCapturesDataset\GT\3.png'), cv2.COLOR_RGB2BGR)

# for img in [noise1, noise2, noise3]:
#     print(is_valid(img))

print(is_valid(noise2))