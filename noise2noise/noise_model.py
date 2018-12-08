import string
import random
import numpy as np

def get_noise_model(noise_type="gaussian,0,50"):
    tokens = noise_type.split(sep=",")

    if tokens[0] == "gaussian":
        min_stddev = int(tokens[1])
        max_stddev = int(tokens[2])

        def gaussian_noise(img):
            noise_img = img.astype(np.float)
            stddev = np.random.uniform(min_stddev, max_stddev)
            noise = np.random.randn(*img.shape) * stddev
            noise_img += noise
            noise_img = np.clip(noise_img, 0, 1).astype(np.float)
            return noise_img
        return gaussian_noise
    elif tokens[0] == "clean":
        return lambda img: img
    else:
        raise ValueError("noise_type should be 'gaussian', 'clean'")

if __name__ == '__main__':
    pass