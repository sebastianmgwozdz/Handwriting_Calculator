import cv2
import numpy as np

class ImageProcessor:
    def contour_boxes(self, path):
        visX = set()
        visY = set()
        boxes = []
        ctrs = self.contours(path)

        rows, cols, ch = cv2.imread(path).shape

        ctrs = sorted(ctrs, key=lambda val: cv2.boundingRect(val)[0])


        for ctr in ctrs:
            x, y, w, h = cv2.boundingRect(ctr)

            if w > cols / 10 or h > rows / 10 and x not in visX and x - 1 not in visX and x + 1 not in visX:
                print((x, y, w, h))

                boxes.append((x, y, w, h))

                visX.add(x)
                visY.add(y)




        return boxes


    def contours(self, path):



        image = cv2.imread(path)

        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 10, 100)

        mser = cv2.MSER_create(_delta=50, _min_diversity=50)

        regions, boundingBoxes = mser.detectRegions(gray)

        cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        return cnts