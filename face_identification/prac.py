import cv2

img = cv2.imread("./img/twice.jpg")

tmp = img[:, 100:200]

cv2.imshow("tmp", tmp)

k = cv2.waitKey(0)

if k == 27:
    cv2.destroyAllWindows()
