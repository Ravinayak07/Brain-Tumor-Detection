import numpy as np
import cv2 as cv


class DisplayDisease:
    curImg = 0
    Img = 0

    def readImage(self, img):
        if isinstance(img, np.ndarray):
            self.Img = np.array(img)
            self.curImg = np.array(img)
            # Update color conversion if needed
            gray = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
            self.ret, self.thresh = cv.threshold(
                gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        else:
            raise ValueError("Input image must be a numpy array")

    def getImage(self):
        return self.curImg

    # noise removal
    def removeNoise(self):
        if self.thresh is not None:
            # Adjust kernel and iterations for morphological operations based on the disease characteristics
            self.kernel = np.ones((3, 3), np.uint8)
            opening = cv.morphologyEx(
                self.thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
            self.curImg = opening
        else:
            raise ValueError("Input image not initialized")

    def displayDisease(self):
        if self.curImg is not None:
            # sure background area
            sure_bg = cv.dilate(self.curImg, self.kernel, iterations=3)

            # Finding sure foreground area
            dist_transform = cv.distanceTransform(self.curImg, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(
                dist_transform, 0.7 * dist_transform.max(), 255, 0)

            # Find unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1

            # Now mark the region of unknown with zero
            markers[unknown == 255] = 0
            markers = cv.watershed(self.Img, markers)
            self.Img[markers == -1] = [255, 0, 0]

            # Update color conversion if needed
            # Update color conversion if needed
            diseaseImage = cv.cvtColor(self.Img, cv.COLOR_HSV2BGR)
            self.curImg = diseaseImage
        else:
            raise ValueError("Input image not initialized")

    def calculateTumorPercentage(self):
        if self.Img is not None:
            gray = cv.cvtColor(self.Img, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
            diseasePercentage = round(
                (cv.countNonZero(binary) / (binary.shape[0] * binary.shape[1])) * 100, 2)
            return diseasePercentage
        else:
            raise ValueError("Input image not initialized")
