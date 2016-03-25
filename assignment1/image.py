"""
Code for working with the image data.
"""
import cv2


class Image:
    """
    A class for storing the image data.
    """
    def __init__(self, image_path):
        self.rgb = cv2.imread(image_path)
        self.grayscale = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY)
