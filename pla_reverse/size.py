"""Functions for dealing with in game height and weight"""

from typing import Union
import numpy as np
from numba_pokemon_prngs.data.personal import PERSONAL_INFO_LA


def float_from_bytes(val: np.uint32) -> np.float32:
    """Convert a u32 representation of a float to a float32"""
    return np.frombuffer(np.uint32(val).tobytes(), dtype=np.float32)[0]


def calc_size_ratio(scalar: np.uint8):
    """Calculate the size ratio from a scalar value"""
    scalar = np.uint8(scalar)
    result = scalar / float_from_bytes(0x437F0000)
    result *= float_from_bytes(0x3ECCCCCE)
    result += float_from_bytes(0x3F4CCCCD)

    return result


def calc_display_size(
    species: np.uint16,
    height_scalar: np.uint8,
    weight_scalar: np.uint8,
    imperial: bool = True,
) -> tuple:
    """Calculate the display sizes from scalar values"""
    height_ratio = calc_size_ratio(height_scalar)
    weight_ratio = calc_size_ratio(weight_scalar)
    ratio = weight_ratio * height_ratio

    height_absolute = np.uint16(PERSONAL_INFO_LA[species].height) * height_ratio
    weight_absolute = np.uint16(PERSONAL_INFO_LA[species].weight) * ratio

    if imperial:
        # TODO: more precise divisors
        # cm -> inches
        height_value = np.round(height_absolute / np.float32(2.54))
        # feet, inches
        height_value = (height_value // 12, height_value % 12)
        # g -> lbs
        weight_value = np.round(weight_absolute / np.float32(4.53592), 1)
    else:
        # cm -> m
        height_value = np.round(height_absolute / np.float32(100), 2)
        # hg -> kg
        weight_value = np.round(weight_absolute / np.float32(10), 2)
    return (height_value, weight_value)


def all_possible_sizes(
    species: np.uint16,
    display_height: Union[np.float32, tuple],
    display_weight: np.float32,
    imperial: bool = True,
) -> tuple:
    """Calculate all possible size combinations"""
    if imperial:
        display_height = np.float32(display_height[0]), np.float32(display_height[1])
    else:
        display_height = np.float32(display_height)
    display_weight = np.float32(display_weight)
    results = []
    for height_scalar in range(0x100):
        for weight_scalar in range(0x100):
            test_height, test_weight = calc_display_size(
                species, height_scalar, weight_scalar, imperial
            )
            if not np.allclose(display_height, test_height):
                continue
            if not np.isclose(display_weight, test_weight):
                continue
            results.append((height_scalar, weight_scalar))
    return tuple(results)


def scalars_to_ushort(height_scalar: np.uint8, weight_scalar: np.uint8):
    """Convert scalars to a single ushort"""
    return (np.uint16(height_scalar) << np.uint16(8)) | np.uint16(weight_scalar)


def build_sizes_table(sizes_set: set) -> tuple:
    """Build sizes table for shader usage"""
    sizes_set_ = {scalars_to_ushort(*size) for size in sizes_set}
    sizes_table = [0 for _ in range(0x10000 // 8)]
    for i in range(0x10000):
        sizes_table[i // 8] |= (1 << (i % 8)) if i in sizes_set_ else 0
    return tuple(sizes_table)
