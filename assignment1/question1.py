"""
Code for question 1. A nonlocal denoising method.
"""
import math
import numpy as np

from image import Image


def find_similar_patches(image, patch_center, patch_size, number_of_similar_patches):
    """
    Finds the most similar patches to a given patch.

    :param image: The image to search for patches in.
    :type image: np.ndarray
    :param patch_center: The coordinates of the patch in question.
    :type patch_center: (int, int)
    :param patch_size: The size of a side of the patch (always square).
    :type patch_size: int
    :param number_of_similar_patches: The number of similar patches to return.
    :type number_of_similar_patches: int
    :return: A list of the coordinates of the similar patch centers.
    :rtype: list[(int, int)]
    """
    # Prepare convenience variables.
    height, width = image.shape
    patch_y, patch_x = patch_center
    patch_edge_offset = math.floor(patch_size / 2)
    query_patch = image[patch_y - patch_edge_offset:patch_y + patch_edge_offset + 1,
                        patch_x - patch_edge_offset:patch_x + patch_edge_offset + 1]
    similar_patches_with_difference = []

    # Iterate over all patches.
    for y_index in range(patch_edge_offset+1, height-patch_edge_offset):
        for x_index in range(patch_edge_offset+1, width-patch_edge_offset):
            if not y_index == patch_y and not x_index == patch_x:
                # Compare patch with query patch.
                comparison_patch = image[y_index - patch_edge_offset:y_index + patch_edge_offset + 1,
                                         x_index - patch_edge_offset:x_index + patch_edge_offset + 1]
                test = np.abs(query_patch.astype(np.int16) - comparison_patch.astype(np.int16))
                sum_difference = np.sum(test)
                similar_patches_with_difference.append((y_index, x_index, sum_difference))

    # Sort and return only the most similar patches.
    top_similar_patches_with_difference = sorted(similar_patches_with_difference,
                                                 key=lambda x: x[2])[:number_of_similar_patches]
    top_similar_patches = [(patch[0], patch[1]) for patch in top_similar_patches_with_difference]
    return top_similar_patches


def nonlocal_mean_denoising(image, patch_size=5, number_of_similar_patches=3):
    """
    Denoises an image using the nonlocal mean approach.

    :param image: The image to search for patches in.
    :type image: np.ndarray
    :param patch_size: The size of a side of the patch (always square).
    :type patch_size: int
    :param number_of_similar_patches: The number of similar patches to return.
    :type number_of_similar_patches: int
    :return: The denoised image.
    :rtype: np.ndarray
    """
    height, width = image.shape
    denoised_image = np.copy(image)
    patch_edge_offset = math.floor(patch_size / 2)

    for y_index in range(patch_edge_offset+1, height-patch_edge_offset):
        for x_index in range(patch_edge_offset+1, width-patch_edge_offset):
            patch_center = (y_index, x_index)
            similar_patch_centers = find_similar_patches(image, patch_center, patch_size, number_of_similar_patches)
            denoised_image[y_index, x_index] = np.mean(image[list(zip(*similar_patch_centers))])


if __name__ == '__main__':
    cameraman_image = Image('cameraman.jpg')
    denoised_cameraman_image = nonlocal_mean_denoising(cameraman_image.grayscale)
