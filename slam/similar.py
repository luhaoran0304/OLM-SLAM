import cv2
import numpy as np


def histogram(image):
    image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    if image_np.ndim == 4:
        image_np = image_np[0]
    image_np = image_np.transpose(1, 2, 0)
    image_np = image_np.astype(np.uint8)
    hist_b = cv2.calcHist([image_np], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_np], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image_np], [2], None, [256], [0, 256])
    hist_all = np.concatenate((hist_b, hist_g, hist_r))
    hist_all_transposed = hist_all.reshape(1, -1)
    return hist_all_transposed


def pHash(image):
    image_np = image.detach().cpu().numpy()
    if image_np.ndim == 4:
        image_np = image_np[0]
    image_np = image_np.transpose(1, 2, 0)
    if image_np.ndim == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    image_np = cv2.resize(image_np, (32, 32), interpolation=cv2.INTER_CUBIC)

    dct = cv2.dct(np.float32(image_np))
    dct_roi = dct[0:8, 0:8]
    average = np.mean(dct_roi)
    hash = []
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash.append(1)
            else:
                hash.append(0)
    hash_np = np.array(hash, dtype=np.float32).reshape(1, -1)
    return hash_np

def cosine(vec1, vec2):
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    normalized_vector1 = vec1 / norm_a
    normalized_vector2 = vec2 / norm_b
    dot_product = np.dot(normalized_vector1, normalized_vector2.T)
    return dot_product

