import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('test_images/test1.jpg')

colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
)

i = 0

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    global i
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        #print(bbox)
        #print(bbox[0], bbox[1])
        # Draw a rectangle given bbox coordinates
        color = colors[i % 3]
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        i += 1
    # Return the image copy with boxes drawn
    return imcopy


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

    x_boundary = x_start_stop[1]
    y_boundary = y_start_stop[1]

    step_sizes = np.array(xy_window) * np.array(xy_overlap)
    x_step = int(step_sizes[0])
    y_step = int(step_sizes[1])

    x_ranges = map(int, x_start_stop + [step_sizes[0]])
    y_ranges = map(int, y_start_stop + [step_sizes[1]])

    x_steps = range(*x_ranges)
    y_steps = range(*y_ranges)

    window_list = []
    for y in y_steps:
        for x in x_steps:
            if x + xy_window[0] > x_boundary or \
                    y + xy_window[1] > y_boundary:
                continue
            window_list.append((
                (x, y),
                (x + xy_window[0], y + xy_window[1])
            ))
        break

    return window_list

from pprint import pprint as pp
windows = slide_window(image, xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#pp(windows)

img = draw_boxes(image, windows)
plt.imsave('windows.jpg', img)
