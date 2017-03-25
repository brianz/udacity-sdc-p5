import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog

def _get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    return hog(img, orientations=orient,
              pixels_per_cell=(pix_per_cell, pix_per_cell),
              cells_per_block=(cell_per_block, cell_per_block),
              transform_sqrt=True,
              visualise=vis, feature_vector=feature_vec)


def get_hog_features(img, channel=0, **kwargs):
    if channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            features = get_hog_features(img[:,:,channel], vis=False, feature_vec=True, **kwargs)
            hog_features.append(features)
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(img[:,:,channel], vis=False, feature_vec=True, **kwargs)

    return hog_features


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    return np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

_spaces = {
    'hsv': cv2.COLOR_RGB2HSV,
    'luv': cv2.COLOR_RGB2LUV,
    'hls': cv2.COLOR_RGB2HLS,
    'yuv': cv2.COLOR_RGB2YUV,
    'ycrcb': cv2.COLOR_RGB2YCrCb,
}

# Define a function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for fn in imgs:
        file_features = []
        img = mpimg.imread(fn)

        cv2_cspace = _spaces.get(cspace.lower())
        feature_image = cv2.cvtColor(img, cv2_cspace) if cv2_cspace else np.copy(img)

        if spatial_feat:
            file_features.append(bin_spatial(feature_image, size=spatial_size))

        if hist_feat:
            file_features.append(color_hist(feature_image, nbins=hist_bins))

        if hog_feat:
            kwargs = {
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
            }
            file_features.append(hog_features(feature_image, hog_channel, **kwargs)

        features.append(np.concatenate(file_features))

    return features


def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)
                ):
    x_window_size, y_window_size = xy_window

    x_step, y_step = map(int, np.array(xy_window) * np.array(xy_overlap))

    if not all(x_start_stop):
        x_start_stop = [0, img.shape[1] - x_window_size + 1]
    if not all(y_start_stop):
        y_start_stop = [0, img.shape[0] - y_window_size + 1]

    x_ranges = map(int, x_start_stop + [x_step])
    y_ranges = map(int, y_start_stop + [y_step])

    x_steps = range(*x_ranges)
    y_steps = range(*y_ranges)

    window_list = []
    for y in y_steps:
        for x in x_steps:
            window_list.append((
                (x, y),
                (x + x_window_size, y + y_window_size)
            ))

    return window_list


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy
