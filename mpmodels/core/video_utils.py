import cv2
import pandas as pd
from typing import Dict


def aspect_preserving_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # taken from: https://stackoverflow.com/a/44659589
    # also used in: https://github.com/PyImageSearch/imutils/blob/c12f15391fcc945d0d644b85194b8c044a392e0a/imutils/convenience.py#L65-L94

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image (note dim is (w,h) following the convention of opencv for resize: https://stackoverflow.com/a/27871067)
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def shortest_side_resize(image, shortest_side_size):
    # cv2 uses numpy for manipulating images, so h, w is the shape order when calling .shape()
    # according to: https://stackoverflow.com/a/19098258
    assert len(image.shape) == 3
    (height, width) = image.shape[:2]
    if width < height:
        image = aspect_preserving_resize(image=image, width=shortest_side_size)
    else:
        image = aspect_preserving_resize(image=image, height=shortest_side_size)
    return image


def label_id_convert(label_to_id_file):
    label_to_id_df = pd.read_csv(label_to_id_file)
    label_to_id_dict: Dict[str, int] = dict()
    id_to_label_dict: Dict[int, str] = dict()
    for index, row in label_to_id_df.iterrows():
        label_to_id_dict[row["name"]] = int(row["id"])
        id_to_label_dict[int(row["id"])] = row["name"]
    return label_to_id_dict, id_to_label_dict


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
        axes_pad=0.3,  # pad between axes in inch.
    )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()
