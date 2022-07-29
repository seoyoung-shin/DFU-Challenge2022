import os
import cv2 as cv
import numpy as np
from PIL import Image
from scipy.stats import truncnorm
from pathlib import Path

def custom_join(first, second):
    return first + "/" + second
RAW_PATH = "c:/thyroid/db/GT_segmentation"
INPUT_PATH = r"C:\AI\DFUC2022\DFUC2022_train\DFUC2022_train_images"
LABEL_PATH = r"C:\AI\DFUC2022\DFUC2022_train\DFUC2022_train_masks_one"

AUGMENTATION_NUMBER = 3
IS_ZOOM_IN_TARGET = True
IS_ROTATE_IN_TARGET = True

MAXIMUM_ANGLE = 15
MAXIMUM_ZOOM_IN = 2
MAXIMUM_ZOOM_OUT = 2
MAXIMUM_BILATERAL_INDEX = 10

OUTPUT_PATH = r"C:\AI\DFUC2022\DFUC2022_train\aug_input"

def get_range_random(low=0, high=1, size=1000):
    mean = 0.5 * (low + high)
    sd = abs(high-low)/5

    return truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd).rvs(size)

def get_half_random(low=0, high=1, size=1000):
    mean = low
    sd = abs(high - low) / 4

    return truncnorm((low - mean) / sd, (high - mean) / sd, loc=mean, scale=sd).rvs(size)

def gamma_correction(src, gamma):
    #gamma는 0 ~ 2
    gamma = min(2, max(0, gamma))
    if gamma < 1:
        gamma = 1 / (2 - gamma)
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def flip_vertical(input_image, value):
    value = min(1, max(0, value))
    output_image = input_image
    if value >= 0.5:
        output_image = cv.flip(input_image, 0)

    return output_image

def flip_horizontal(input_image, value):
    value = min(1, max(0, value))
    output_image = input_image
    if value >= 0.5:
        output_image = cv.flip(input_image, 1)

    return output_image

def rotate_image(src, angle, center_x, center_y, is_image=True):
    angle = min(90, max(-90, angle))

    image_PIL = Image.fromarray(src)
    if is_image:
        output_PIL = image_PIL.rotate(angle, resample=Image.BILINEAR, expand=False, center=(center_x, center_y))
    else:
        output_PIL = image_PIL.rotate(angle, resample=Image.NEAREST, expand=False, center=(center_x, center_y))

    output = np.array(output_PIL)
    return output

def rotate_image_on_center(src, angle, is_image=True):
    angle = min(90, max(-90, angle))

    center_x = int(0.5 * src.shape[1] + 0.5)
    center_y = int(0.5 * src.shape[0] + 0.5)

    return rotate_image(src, angle, center_x, center_y, is_image)

def bilateral_filter(input_image, value):
    value = min(100, max(0, value))
    return cv.bilateralFilter(input_image, -1, value, value)

def shapren_filter(input_image, value):
    value = min(1, max(0, value))
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    image_sharp = cv.filter2D(src=input_image, ddepth=-1, kernel=kernel)
    output_image = cv.addWeighted(input_image, (1 - value), image_sharp, value, 0)
    return output_image

def zoom_out_sliding(input_image, value, position_x=-1, position_y=-1, is_image=True):
    value = max(1, value)
    output_image = np.zeros(input_image.shape, dtype=np.uint8)
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]

    if is_image:
        zoom_image = cv.resize(input_image, (0, 0), fx=1/value, fy=1/value, interpolation=cv.INTER_LINEAR)
    else:
        zoom_image = cv.resize(input_image, (0, 0), fx=1 / value, fy=1 / value, interpolation=cv.INTER_NEAREST)
    width_zoom = zoom_image.shape[1]
    height_zoom = zoom_image.shape[0]

    if position_x < 0:
        if width_input == width_zoom:
            position_x = 0
        else:
            position_x = np.random.randint(0, (width_input - width_zoom))
    if position_y < 0:
        if height_input == height_zoom:
            position_y = 0
        else:
            position_y = np.random.randint(0, (height_input - height_zoom))

    output_image[position_y:position_y + height_zoom, position_x:position_x + width_zoom] = zoom_image

    return output_image, position_x, position_y

def zoom_in_position(input_image, value, position_x, position_y, is_image=True):
    value = max(1, value)
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]
    width_input_half = int(0.5 * width_input + 0.5)
    height_input_half = int(0.5 * height_input + 0.5)

    if is_image:
        zoom_image = cv.resize(input_image, (0, 0), fx=value, fy=value, interpolation=cv.INTER_LINEAR)
    else:
        zoom_image = cv.resize(input_image, (0, 0), fx=value, fy=value, interpolation=cv.INTER_NEAREST)

    width_zoom = zoom_image.shape[1]
    height_zoom = zoom_image.shape[0]

    zoom_position_x = int(position_x * value + 0.5)
    zoom_position_y = int(position_y * value + 0.5)

    min_x = max(0, zoom_position_x - width_input_half)
    min_y = max(0, zoom_position_y - height_input_half)
    max_x = min(width_zoom, min_x + width_input)
    max_y = min(height_zoom, min_y + height_input)
    if max_x == width_zoom:
        min_x = max_x - width_input
    if max_y == height_zoom:
        min_y = max_y - height_input
    output_image = zoom_image[min_y : max_y, min_x : max_x]

    return output_image

def calculate_roi_from_contour(input_contour, width, height):
    x, y, w, h = cv.boundingRect(input_contour)
    x_min = x
    y_min = y
    x_max = x + w
    y_max = y + h

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    return [x_min, y_min, x_max, y_max]

def zoom_in_center(input_image, value, is_image=True):
    width_input = input_image.shape[1]
    height_input = input_image.shape[0]

    point_center_x = int(0.5 * width_input + 0.5)
    point_center_y = int(0.5 * height_input + 0.5)

    return zoom_in_position(input_image, value, point_center_x, point_center_y, is_image)

def find_largest_external_contour(input_contours):
    max_area = 0
    result_contour = []
    for current_contour in input_contours:
        current_area = cv.contourArea(current_contour)
        if (current_area >= max_area):
            max_area = current_area
            result_contour = current_contour
    return result_contour

def get_segmentation_center(label_image):
    contours, _ = cv.findContours(label_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    largest_contour = find_largest_external_contour(contours)
    M = cv.moments(largest_contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    return center_x, center_y

def change_path(original_path, from_name, to_name):
    output_name = original_path.replace(from_name, to_name)
    path = Path(output_name).parent
    if not os.path.exists(path):
        os.makedirs(path)

    return output_name

# with open(custom_join(INPUT_PATH, "image_list.txt"), "r") as file:
#     image_file_list = file.readlines()
# with open(custom_join(LABEL_PATH, "label_list.txt"), "r") as file:
#     label_file_list = file.readlines()

image_file_list = []
label_file_list = []
image_list_dir = os.listdir(INPUT_PATH)
for current_name in image_list_dir:
    current_image = os.path.join(INPUT_PATH, current_name)
    current_label = os.path.join(LABEL_PATH, current_name.replace(".jpg", ".png"))

    image_file_list.append(current_image)
    label_file_list.append(current_label)

file_count = len(image_file_list)
augmentation_count = file_count * AUGMENTATION_NUMBER
#Geometric Augmentation (Image and Label)
value_v_flip = np.random.uniform(low=0, high=1, size=augmentation_count)
value_h_flip = np.random.uniform(low=0, high=1, size=augmentation_count)
value_rotation = get_range_random(low=-MAXIMUM_ANGLE , high=MAXIMUM_ANGLE , size=augmentation_count) # center based
value_zoom_in = get_half_random(low=1, high=MAXIMUM_ZOOM_IN , size=augmentation_count) # zoom-in (ROI center)
value_zoom_out = get_half_random(low=1, high=MAXIMUM_ZOOM_OUT , size=augmentation_count) # center based zoom-out and random position
#Graphic Augmentation (Only Image)
value_sharpen = get_half_random(low=0, high=1, size=augmentation_count)
value_bilateral = get_half_random(low=0, high=MAXIMUM_BILATERAL_INDEX , size=augmentation_count) # image filter
value_gamma = get_range_random(low=0, high=2, size=augmentation_count) #for contrast and brightness (0.5 ~ 2로...) 1 ~ 2로하고, 1/1 ~ 1/2까지..(-1에서 1로하고... 변환에서 쓸까)

for index_augmentation in range(0, AUGMENTATION_NUMBER):
    output_path = custom_join(OUTPUT_PATH, "train_aug_" + str(index_augmentation+1))
    output_image_path = os.path.join(output_path, "image")
    output_label_path = os.path.join(output_path, "mask")
    output_label_debug_path = os.path.join(output_path, "mask_debug")
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    if not os.path.exists(output_label_path):
        os.makedirs(output_label_path)
    if not os.path.exists(output_label_debug_path):
        os.makedirs(output_label_debug_path)

    file_pointer_input = open(custom_join(output_image_path, "image_list.txt"), "w")
    file_pointer_label = open(custom_join(output_label_path, "label_list.txt"), "w")
    print("augmentation count = " + str(index_augmentation) + "...")
    for index_data in range(0, file_count):
        index = (index_augmentation + 1) * index_data

        input_image_name = image_file_list[index_data].replace("\n", "")
        input_label_name = label_file_list[index_data].replace("\n", "")
        output_image_name = change_path(input_image_name, INPUT_PATH, output_image_path)
        output_label_name = change_path(input_label_name, LABEL_PATH, output_label_path)
        output_label_debug_name  = change_path(input_label_name, LABEL_PATH, output_label_debug_path)
        print(input_image_name)

        input_image = cv.imread(input_image_name)
        input_label = cv.imread(input_label_name, cv.IMREAD_GRAYSCALE)

        #Geometric Augmentation
        image_v_flip = flip_vertical(input_image, value_v_flip[index])
        label_v_flip = flip_vertical(input_label, value_v_flip[index])
        image_h_flip = flip_horizontal(image_v_flip, value_h_flip[index])
        label_h_flip = flip_horizontal(label_v_flip, value_h_flip[index])
        label_center_x, label_center_y = get_segmentation_center(label_h_flip)
        if IS_ROTATE_IN_TARGET:
            image_rotation = rotate_image(image_h_flip, value_rotation[index], label_center_x, label_center_y)
            label_rotation = rotate_image(label_h_flip, value_rotation[index], label_center_x, label_center_y, False)
        else:
            image_rotation = rotate_image_on_center(image_h_flip, value_rotation[index])
            label_rotation = rotate_image_on_center(label_h_flip, value_rotation[index], False)
        label_center_x, label_center_y = get_segmentation_center(label_rotation)
        if IS_ZOOM_IN_TARGET:
            image_zoom_in = zoom_in_position(image_rotation, value_zoom_in[index], label_center_x, label_center_y)
            label_zoom_in = zoom_in_position(label_rotation, value_zoom_in[index], label_center_x, label_center_y, False)
        else:
            image_zoom_in = zoom_in_center(image_rotation, value_zoom_in[index])
            label_zoom_in = zoom_in_center(label_rotation, value_zoom_in[index], False)

        image_zoom_out, zoom_out_center_x, zoom_out_center_y = zoom_out_sliding(image_zoom_in, value_zoom_out[index])
        output_label, _, _ = zoom_out_sliding(label_zoom_in, value_zoom_out[index], zoom_out_center_x, zoom_out_center_y, False)

        #Graphic Augmentation
        image_sharpen = shapren_filter(image_zoom_out, value_sharpen[index])
        image_bilateral = bilateral_filter(image_sharpen, value_bilateral[index])
        output_image = gamma_correction(image_bilateral, value_gamma[index])

        file_pointer_input.write(output_image_name + "\n")
        file_pointer_label.write(output_label_name + "\n")

        cv.imwrite(output_image_name, output_image)
        cv.imwrite(output_label_name, output_label)
        cv.imwrite(output_label_debug_name, output_label * 255)
    file_pointer_input.close()
    file_pointer_label.close()

print("finish")


