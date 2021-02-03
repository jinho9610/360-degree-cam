from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw
import numpy as np
import cv2

# draw an image with detected objects


def figure_to_array(fig):
    """
    plt.figure를 RGBA로 변환(layer가 4개)
    shape: height, width, layer
    """
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)


def draw_image_with_boxes(img, result_list):
    # load the image
    # 이미 data가 ndarray 즉, img인 상태
    # plot the image
    # get the context for drawing boxes
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)
        # draw the box
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            cv2.circle(img, (value[0], value[1]), 2, (0, 0, 255), -1)
    # show the plot
    # pyplot.show()
    return img


if __name__ == '__main__':
    detector = MTCNN()

    input_video = cv2.VideoCapture('360degree.mp4')
    ret, img = input_video.read()
    h, w, c = img.shape
    length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ouput_video = cv2.VideoWriter('output.avi', fourcc, 29.97, (w, h))

    ret = True
    frame_number = 1
    while ret:
        ret, frame = input_video.read()  # frame 하나씩 읽기

        faces = detector.detect_faces(frame)
        ouput_video.write(draw_image_with_boxes(frame, faces))

        print("Writing frame {} / {}".format(frame_number, length))
        frame_number += 1
