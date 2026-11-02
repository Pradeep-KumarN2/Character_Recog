import os
import cv2 as cv
import numpy as np
import concurrent.futures
from typing import Callable, Any

def mean_filter(image: np.ndarray, kernel_size: tuple[int,int]=(5,5)) -> np.ndarray:
    image = cv.blur(image, kernel_size)
    return image

def gaussian_filter(image: np.ndarray, kernel_size: tuple[int,int]=(5,5)) -> np.ndarray:
    image = cv.GaussianBlur(image, kernel_size, 0)
    return image


def binarize_image(
    image: np.ndarray, kernel_size: tuple[int, int] = (5, 5)
) -> np.ndarray:
    image = cv.GaussianBlur(image, kernel_size, 0)
    _, image = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return image


def crop_image(image: np.ndarray) -> np.ndarray:
    negative_image = cv.bitwise_not(image)
    contours, _ = cv.findContours(negative_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    x_max, y_max = 0, 0
    x_min, y_min = image.shape
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x + w, x_max)
        y_max = max(y + h, y_max)
    return image[y_min:y_max, x_min:x_max]


def resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    
    return cv.resize(image, shape, cv.INTER_NEAREST)


def sift_shading(image: np.ndarray, grid_size: int = 8) -> np.ndarray:
    h, w = image.shape
    for x in range(0, h, grid_size):
        for y in range(0, w, grid_size):
            c = image[x : x + grid_size, y : y + grid_size].ravel()
            if len(c) == grid_size**2 and all(c == 255):
                c[0 : len(c) // 2] = 0
                image[x : x + grid_size, y : y + grid_size] = c.reshape(
                    grid_size, grid_size
                )
    return image


class ImageDirPreprocessingPipeline:
    def __init__(
        self,
        image_dir: str,
        out_dir: str,
        save_as: str,
        preprocess_steps: list[tuple[Callable[[np.ndarray, Any], np.ndarray]]],
    ) -> None:
        self.image_dir = image_dir
        self.out_dir = out_dir
        self.save_as = save_as
        self.preprocess_steps = preprocess_steps

    def _transform_io(
        self,
        image_path: str,
    ) -> None:
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        image_name = os.path.basename(image_path)
        basename, _ = os.path.splitext(os.path.join(self.out_dir, image_name))
        for transformation, *args in self.preprocess_steps:
            image = transformation(image, *args)
        cv.imwrite(f'{basename}.{self.save_as}', image)

    def transform(self) -> None:
        # output directory does not exists
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        image_paths = [
            os.path.join(self.image_dir, image_name)
            for image_name in os.listdir(self.image_dir)
        ]
        
        # passing each image through pipeline concurrenlty
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._transform_io, image_paths)
