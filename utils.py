import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read image to RGB format.
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def show_image(img, figsize=(10, 10), gray=False):
    plt.figure(figsize=figsize)
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()


def rgb_to_yuv(rgb0):
    rgb = rgb0 / 255.0
    y = np.clip(np.dot(rgb, np.array([0.299, 0.587, 0.144])), 0,   1)
    i = np.clip(np.dot(rgb, np.array([0.595716, -0.274453, -0.321263])), -0.5957, 0.5957)
    q = np.clip(np.dot(rgb, np.array([0.211456, -0.522591, 0.311135])), -0.5226, 0.5226)
    yiq = rgb
    yiq[..., 0] = y
    yiq[..., 1] = i
    yiq[..., 2] = q
    return yiq


def yuv_to_rgb(yuv):
    yiq = yuv.copy()
    r = np.dot(yiq, np.array([1.0,  0.956295719758948,  0.621024416465261]))
    g = np.dot(yiq, np.array([1.0, -0.272122099318510, -0.647380596825695]))
    b = np.dot(yiq, np.array([1.0, -1.106989016736491,  1.704614998364648]))
    rgb = yiq
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.clip(rgb, 0.0, 1.0) * 255.0


if __name__ == '__main__':
    img1 = read_image('data/baby.bmp')
    print(img1.shape, img1.dtype)
    # img2 = read_image('data/baby_marked.bmp')
    # print(img2.shape)
    # m, n, _ = img1.shape
    # cnt = 0
    # for i in range(m):
    #     for j in range(n):
    #         if (img1[i, j] == img2[i, j]).sum() != 3:
    #             cnt += 1
    # print(cnt, m * n)
    yuv = rgb_to_yuv(img1)
    print(yuv.shape)
    rgb = yuv_to_rgb(yuv)
    print(np.allclose(img1, rgb.astype(np.uint8)))
    print(img1)
    print(rgb.astype(np.uint8))
    show_image(img1)
    show_image(rgb.astype(np.uint8))
