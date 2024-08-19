import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
from pylab import *


def histogram_equalization_method():
    color_img = cv2.imread('car.png')

    b, g, r = cv2.split(color_img)

    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)


    equalized_color_img = cv2.merge((eq_b, eq_g, eq_r))

    equalized_hist_r = cv2.calcHist([eq_r], [0], None, [256], [0, 256])
    equalized_hist_g = cv2.calcHist([eq_g], [0], None, [256], [0, 256])
    equalized_hist_b = cv2.calcHist([eq_b], [0], None, [256], [0, 256])


    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')

    axes[0, 1].plot(hist_r, color='r')
    axes[0, 1].plot(hist_g, color='g')
    axes[0, 1].plot(hist_b, color='b')
    axes[0, 1].set_title('Histogram of Original Image')
    axes[0, 1].set_axisbelow(True)
    axes[0, 1].grid(True, linestyle='--', zorder=0)
    axes[1, 0].imshow(cv2.cvtColor(equalized_color_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Equalized Color Image')

    axes[1, 1].plot(equalized_hist_r, color='r')
    axes[1, 1].plot(equalized_hist_g, color='g')
    axes[1, 1].plot(equalized_hist_b, color='b')
    axes[1, 1].set_title('Histogram of Equalized Color Image')
    axes[1, 1].set_axisbelow(True)

    plt.show()

def change_canel_color():

    change_canel_color_img = cv2.imread('car.png')
    change_canel_color_img = cv2.cvtColor(change_canel_color_img, cv2.COLOR_BGR2YCrCb)
 
    b, g, r = cv2.split(change_canel_color_img)

    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

    eq_b = cv2.equalizeHist(b)
    eq_g = cv2.equalizeHist(g)
    eq_r = cv2.equalizeHist(r)

    equalized_color_img = cv2.merge((eq_b, eq_g, eq_r))

    equalized_hist_r = cv2.calcHist([eq_r], [0], None, [256], [0, 256])
    equalized_hist_g = cv2.calcHist([eq_g], [0], None, [256], [0, 256])
    equalized_hist_b = cv2.calcHist([eq_b], [0], None, [256], [0, 256])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(change_canel_color_img)
    axes[0, 0].set_title('Original Image')

    axes[0, 1].plot(hist_r, color='r')
    axes[0, 1].plot(hist_g, color='g')
    axes[0, 1].plot(hist_b, color='b')
    axes[0, 1].set_title('Histogram of Original Image')
    axes[0, 1].set_axisbelow(True)
    axes[0, 1].grid(True, linestyle='--', zorder=0)
    axes[1, 0].imshow(equalized_color_img)
    axes[1, 0].set_title('Equalized Color Image')

    axes[1, 1].plot(equalized_hist_r, color='r')
    axes[1, 1].plot(equalized_hist_g, color='g')
    axes[1, 1].plot(equalized_hist_b, color='b')
    axes[1, 1].set_title('Histogram of Equalized Color Image')
    axes[1, 1].set_axisbelow(True)

    plt.show()

def gamma_correction_1():
    img = cv2.imread('car.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gamma_corrected = exposure.adjust_gamma(img,1)
    gamma_corrected1 = exposure.adjust_gamma(img,0.2)
    gamma_corrected2 = exposure.adjust_gamma(img,0.5)

    r = np.arange(0,256)
    y = np.power(r,1)

    r1 = np.arange(0,256)
    y1 = np.power(r,0.2)

    r2 = np.arange(0,256)
    y2 = np.power(r,0.5)


    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (10, 15))

    axs[0, 0].imshow(gamma_corrected)
    axs[0, 0].set_title('Gamma_with_1 Image')
    axs[1, 0].imshow(gamma_corrected1)
    axs[1, 0].set_title('Gamma_with_0.2 Image')
    axs[2, 0].imshow(gamma_corrected2)
    axs[2, 0].set_title('Gamma_with_0.5 Image')

    axs[0, 1].plot(r,y, color='b')
    axs[0, 1].set_title('1_Gamma Histogram')
    axs[1, 1].plot(r1,y1, color='b')
    axs[1, 1].set_title('0.2_Gamma Histogram')
    axs[2, 1].plot(r2,y2, color='b')
    axs[2, 1].set_title('0.5_Gamma Histogram')

    plt.show()

def gamma_correction_2():
    img = cv2.imread('car.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    gamma_corrected = exposure.adjust_gamma(img,1)
    gamma_corrected3 = exposure.adjust_gamma(img,2)
    gamma_corrected4 = exposure.adjust_gamma(img,5)

    r = np.arange(0,256)
    y = np.power(r,1)

    r3 = np.arange(0,256)
    y3 = np.power(r,2)

    r4 = np.arange(0,256)
    y4 = np.power(r,5)
    fig, axs = plt.subplots(nrows = 3, ncols = 2, figsize = (10, 15))
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Gamma_with_1 Image')
    axs[1, 0].imshow(gamma_corrected3)
    axs[1, 0].set_title('Gamma_with_2 Image')
    axs[2, 0].imshow(gamma_corrected4)
    axs[2, 0].set_title('Gamma_with_5 Image')

    axs[0, 1].plot(r,y, color='b')
    axs[0, 1].set_title('1_Gamma Histogram')
    axs[1, 1].plot(r3,y3, color='b')
    axs[1, 1].set_title('2_Gamma Histogram')
    axs[2, 1].plot(r4,y4, color='b')
    axs[2, 1].set_title('5_Gamma Histogram')
    plt.show()

def contrast_stretching_color_image_function(img, a, b, c, d):
        stretched_img = np.zeros_like(img)

        for channel in range(img.shape[2]):
            pixel_vals = img[:, :, channel]
            stretched_vals = np.where(pixel_vals < a, c, np.where(pixel_vals > b, d, ((pixel_vals - a) * ((d - c) / (b - a))) + c))
            stretched_img[:, :, channel] = stretched_vals

        return stretched_img

def contrast_stretching_method():
    img = cv2.imread('car.png', cv2.IMREAD_COLOR)

    # Convert the image to RGB for displaying with Matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calculate the histograms of the original and contrast-stretched images
    orig_hist_r, _ = np.histogram(img[:, :, 0], bins=256, range=(0, 255))
    orig_hist_g, _ = np.histogram(img[:, :, 1], bins=256, range=(0, 255))
    orig_hist_b, _ = np.histogram(img[:, :, 2], bins=256, range=(0, 255))

    stretched_img = contrast_stretching_color_image_function(img, 50, 200, 0, 255)
    stretched_hist_r, _ = np.histogram(stretched_img[:, :, 0], bins=256, range=(0, 255))
    stretched_hist_g, _ = np.histogram(stretched_img[:, :, 1], bins=256, range=(0, 255))
    stretched_hist_b, _ = np.histogram(stretched_img[:, :, 2], bins=256, range=(0, 255))

    # Plot the images and histograms
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axs[0, 0].imshow(img)
    axs[0, 0].set_title('Original Image')

    axs[1, 0].imshow(stretched_img)
    axs[1, 0].set_title('Contrast Stretching Image')

    axs[0, 1].plot(orig_hist_r, color='red')
    axs[0, 1].plot(orig_hist_g, color='green')
    axs[0, 1].plot(orig_hist_b, color='blue')
    axs[0, 1].set_xlim([0, 255])
    axs[0, 1].set_ylim([0, max(max(orig_hist_r), max(orig_hist_g), max(orig_hist_b)) + 100])
    axs[0, 1].set_title('Histogram of Original Image')

    axs[1, 1].plot(stretched_hist_r, color='red')
    axs[1, 1].plot(stretched_hist_g, color='green')
    axs[1, 1].plot(stretched_hist_b, color='blue')
    axs[1, 1].set_xlim([0, 255])
    axs[1, 1].set_ylim([0, max(max(stretched_hist_r), max(stretched_hist_g), max(stretched_hist_b)) + 100])
    axs[1, 1].set_title('Histogram of Contrast Stretching Image')

    plt.show()
contrast_stretching_method()