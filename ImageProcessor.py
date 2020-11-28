import cv2
import numpy as np

class ImageProcessor:
    def contour_boxes(self, path):
        visX = set()
        visY = set()
        boxes = []
        ctrs = self.contours(path)

        rows, cols, ch = cv2.imread(path).shape



        for i, ctr in enumerate(ctrs):
            x, y, w, h = ctr


            if w > cols / 8 and h > rows / 8 and x not in visX and x - 1 not in visX and x + 1 not in visX:
                print((x, y, w, h))

                boxes.append((x, y, w, h))

                visX.add(x)
                visY.add(y)




        return boxes


    def contours(self, path):



        image = cv2.imread(path)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mser = cv2.MSER_create(_delta=25)

        regions, boundingBoxes = mser.detectRegions(gray)

        return boundingBoxes