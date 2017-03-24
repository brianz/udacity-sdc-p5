import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test1.jpg')

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        #print(bbox)
        #print(bbox[0], bbox[1])
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

expected = [((0, 0), (128, 128)), ((64, 0), (192, 128)), ((128, 0), (256, 128)), ((192, 0), (320, 128)), ((256, 0), (384, 128)), ((320, 0), (448, 128)), ((384, 0), (512, 128)), ((448, 0), (576, 128)), ((512, 0), (640, 128)), ((576, 0), (704, 128)), ((640, 0), (768, 128)), ((704, 0), (832, 128)), ((768, 0), (896, 128)), ((832, 0), (960, 128)), ((896, 0), (1024, 128)), ((960, 0), (1088, 128)), ((1024, 0), (1152, 128)), ((1088, 0), (1216, 128)), ((1152, 0), (1280, 128)), ((0, 64), (128, 192)), ((64, 64), (192, 192)), ((128, 64), (256, 192)), ((192, 64), (320, 192)), ((256, 64), (384, 192)), ((320, 64), (448, 192)), ((384, 64), (512, 192)), ((448, 64), (576, 192)), ((512, 64), (640, 192)), ((576, 64), (704, 192)), ((640, 64), (768, 192)), ((704, 64), (832, 192)), ((768, 64), (896, 192)), ((832, 64), (960, 192)), ((896, 64), (1024, 192)), ((960, 64), (1088, 192)), ((1024, 64), (1152, 192)), ((1088, 64), (1216, 192)), ((1152, 64), (1280, 192)), ((0, 128), (128, 256)), ((64, 128), (192, 256)), ((128, 128), (256, 256)), ((192, 128), (320, 256)), ((256, 128), (384, 256)), ((320, 128), (448, 256)), ((384, 128), (512, 256)), ((448, 128), (576, 256)), ((512, 128), (640, 256)), ((576, 128), (704, 256)), ((640, 128), (768, 256)), ((704, 128), (832, 256)), ((768, 128), (896, 256)), ((832, 128), (960, 256)), ((896, 128), (1024, 256)), ((960, 128), (1088, 256)), ((1024, 128), (1152, 256)), ((1088, 128), (1216, 256)), ((1152, 128), (1280, 256)), ((0, 192), (128, 320)), ((64, 192), (192, 320)), ((128, 192), (256, 320)), ((192, 192), (320, 320)), ((256, 192), (384, 320)), ((320, 192), (448, 320)), ((384, 192), (512, 320)), ((448, 192), (576, 320)), ((512, 192), (640, 320)), ((576, 192), (704, 320)), ((640, 192), (768, 320)), ((704, 192), (832, 320)), ((768, 192), (896, 320)), ((832, 192), (960, 320)), ((896, 192), (1024, 320)), ((960, 192), (1088, 320)), ((1024, 192), (1152, 320)), ((1088, 192), (1216, 320)), ((1152, 192), (1280, 320)), ((0, 256), (128, 384)), ((64, 256), (192, 384)), ((128, 256), (256, 384)), ((192, 256), (320, 384)), ((256, 256), (384, 384)), ((320, 256), (448, 384)), ((384, 256), (512, 384)), ((448, 256), (576, 384)), ((512, 256), (640, 384)), ((576, 256), (704, 384)), ((640, 256), (768, 384)), ((704, 256), (832, 384)), ((768, 256), (896, 384)), ((832, 256), (960, 384)), ((896, 256), (1024, 384)), ((960, 256), (1088, 384)), ((1024, 256), (1152, 384)), ((1088, 256), (1216, 384)), ((1152, 256), (1280, 384)), ((0, 320), (128, 448)), ((64, 320), (192, 448)), ((128, 320), (256, 448)), ((192, 320), (320, 448)), ((256, 320), (384, 448)), ((320, 320), (448, 448)), ((384, 320), (512, 448)), ((448, 320), (576, 448)), ((512, 320), (640, 448)), ((576, 320), (704, 448)), ((640, 320), (768, 448)), ((704, 320), (832, 448)), ((768, 320), (896, 448)), ((832, 320), (960, 448)), ((896, 320), (1024, 448)), ((960, 320), (1088, 448)), ((1024, 320), (1152, 448)), ((1088, 320), (1216, 448)), ((1152, 320), (1280, 448)), ((0, 384), (128, 512)), ((64, 384), (192, 512)), ((128, 384), (256, 512)), ((192, 384), (320, 512)), ((256, 384), (384, 512)), ((320, 384), (448, 512)), ((384, 384), (512, 512)), ((448, 384), (576, 512)), ((512, 384), (640, 512)), ((576, 384), (704, 512)), ((640, 384), (768, 512)), ((704, 384), (832, 512)), ((768, 384), (896, 512)), ((832, 384), (960, 512)), ((896, 384), (1024, 512)), ((960, 384), (1088, 512)), ((1024, 384), (1152, 512)), ((1088, 384), (1216, 512)), ((1152, 384), (1280, 512)), ((0, 448), (128, 576)), ((64, 448), (192, 576)), ((128, 448), (256, 576)), ((192, 448), (320, 576)), ((256, 448), (384, 576)), ((320, 448), (448, 576)), ((384, 448), (512, 576)), ((448, 448), (576, 576)), ((512, 448), (640, 576)), ((576, 448), (704, 576)), ((640, 448), (768, 576)), ((704, 448), (832, 576)), ((768, 448), (896, 576)), ((832, 448), (960, 576)), ((896, 448), (1024, 576)), ((960, 448), (1088, 576)), ((1024, 448), (1152, 576)), ((1088, 448), (1216, 576)), ((1152, 448), (1280, 576)), ((0, 512), (128, 640)), ((64, 512), (192, 640)), ((128, 512), (256, 640)), ((192, 512), (320, 640)), ((256, 512), (384, 640)), ((320, 512), (448, 640)), ((384, 512), (512, 640)), ((448, 512), (576, 640)), ((512, 512), (640, 640)), ((576, 512), (704, 640)), ((640, 512), (768, 640)), ((704, 512), (832, 640)), ((768, 512), (896, 640)), ((832, 512), (960, 640)), ((896, 512), (1024, 640)), ((960, 512), (1088, 640)), ((1024, 512), (1152, 640)), ((1088, 512), (1216, 640)), ((1152, 512), (1280, 640)), ((0, 576), (128, 704)), ((64, 576), (192, 704)), ((128, 576), (256, 704)), ((192, 576), (320, 704)), ((256, 576), (384, 704)), ((320, 576), (448, 704)), ((384, 576), (512, 704)), ((448, 576), (576, 704)), ((512, 576), (640, 704)), ((576, 576), (704, 704)), ((640, 576), (768, 704)), ((704, 576), (832, 704)), ((768, 576), (896, 704)), ((832, 576), (960, 704)), ((896, 576), (1024, 704)), ((960, 576), (1088, 704)), ((1024, 576), (1152, 704)), ((1088, 576), (1216, 704)), ((1152, 576), (1280, 704))]

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)
                ):
    # If x and/or y start/stop positions not defined, set to image size
    if not all(x_start_stop):
        x_start_stop = [0, img.shape[1]]
    if not all(y_start_stop):
        y_start_stop = [0, img.shape[0]]

    step_sizes = np.array(xy_window) * np.array(xy_overlap)
    x_step = int(step_sizes[0])
    y_step = int(step_sizes[1])

    x_ranges = map(int, x_start_stop + [step_sizes[0]])
    y_ranges = map(int, y_start_stop + [step_sizes[1]])

    print(list(x_ranges))
    x_steps = range(*x_ranges)


    y_steps = range(*y_ranges)

    return
    window_list = []

    for y in y_steps:
        for x in x_steps:
            window_list.append((
                (x, y),
                (x + xy_window[0], y + xy_window[1])
            ))

    return window_list

windows = slide_window(image, xy_window=(128, 128), xy_overlap=(0.5, 0.5))
