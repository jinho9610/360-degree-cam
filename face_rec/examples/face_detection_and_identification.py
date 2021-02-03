from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
import cv2

# draw an image with detected objects


def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        print(x, y, width, height)
        pyplot.imshow(data)
        pyplot.show()
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            # key 목록 : left_eye, right_eye, nose, mouth_left, mouth_right
            # value는 (x, y) 형태
            dot = Circle(value, radius=2, color='blue')  # 눈코입 점
            ax.add_patch(dot)
    # show the plot
    pyplot.show()


def extract_face(filename, result_list):  # 영상에서 얼굴인식해서 얼굴만 잘라내는 함수
    # load the image
    img = cv2.imread(filename)

    idx = 1
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        result = img[y: y + height, x: x + width]
        cv2.imwrite('./img/face' + str(idx) + '.png', result)

        idx += 1


if __name__ == '__main__':
    filename = 'sharon_stone1.jpg'
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    #draw_image_with_boxes(filename, faces)
    extract_face(filename, faces)
