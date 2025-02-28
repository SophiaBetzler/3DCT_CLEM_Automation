import csv
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile as tff
import yaml
from ome_types import from_tiff
from ome_types.model.simple_types import UnitsLength
from PIL import Image

############# PARSER FUNCTIONS #############

def parse_coordinates(fib_coord_filename: str, fm_coord_filename: str) -> list:
    """Parse the coordinates from the old style coordinate files"""

    def parse_coordinate_file(filename: str, delimiter: str = "\t") -> list:
        coords: list = []
        with open(filename) as csv_file:
            for row in csv.reader(csv_file, delimiter=delimiter):
                coords.append([field for field in row])
        return coords

    fib_coordinates = parse_coordinate_file(fib_coord_filename)
    fm_coordinates = parse_coordinate_file(fm_coord_filename)

    fib_coordinates = np.array(fib_coordinates, dtype=np.float32)
    fm_coordinates = np.array(fm_coordinates, dtype=np.float32)

    return fib_coordinates, fm_coordinates

def parse_metadata(filename: str) -> np.ndarray:
    """parse metadata from a tfs tiff file"""
    # TODO: replace this with real parser versions eventually
    md = {}
    with tff.TiffFile(filename) as tif:
        for page in tif.pages:
            for tag in page.tags.values():
                if tag.name == "FEI_HELIOS" or tag.code == 34682: # TFS_MD tag
                    md = tag.value
    return md

def load_image_and_metadata(filename: str) -> tuple[np.ndarray, dict]:
    # TODO: convert to FIBSEMImage always, require the package...
    try:
        from fibsem.structures import FibsemImage
        image = FibsemImage.load(filename)
        pixel_size = image.metadata.pixel_size.x
        image = image.data
    except Exception as e:
        logging.debug(f"Failed to load as FibsemImage: {e}")

        try:
            image, pixel_size = load_tfs_image(filename)
        except Exception as e:
            logging.error(f"Failed to load as TFS image: {e}")
            return None, None

    return image, pixel_size

def load_tfs_image(filename: str) -> Tuple[np.ndarray, float]:
    """Load a TFS image and extract the pixel size from the metadata"""
    image = tff.imread(filename)
    md = parse_metadata(filename)

    pixel_size = None
    try:
        pixel_size = md["Scan"]["PixelWidth"]
    except KeyError as e:
        logging.warning(f"Pixel size not found in metadata: {e}")
        pass

    # convert to grayscale
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image).convert("L"))

    trim_metadata: bool = False
    try:
        shape = md["Image"]["ResolutionY"], md["Image"]["ResolutionX"]
        if image.shape != shape:
            logging.info(
                f"Image shape {image.shape} does not match metadata shape {shape}, likely a metadata bar present"
            )
            trim_metadata = True
    except KeyError as e:
        logging.warning(f"Image shape not found in metadata: {e}")
        pass

    # trim the image to before the first row with all zeros
    if trim_metadata:
        try:
            # crop the image to the metadata bar
            cropped_img = image[: shape[0], : shape[1]]
            # remove the metadata bar with image processing
            trimmed_img = remove_metadata_bar(image)

            logging.info(
                f"Cropped Shape: {cropped_img.shape}, Trimmed Shape: {trimmed_img.shape}"
            )
            if cropped_img.shape != trimmed_img.shape:
                raise ValueError(
                    "Cropped image shape does not match trimmed image shape"
                )

            if image.shape != trimmed_img.shape:
                logging.info(f"Image trimmed from {image.shape} to {trimmed_img.shape}")
                image = trimmed_img
        except Exception as e:
            logging.error(f"Error trimming image: {e}")
            pass

    return image, pixel_size

def remove_metadata_bar(img: np.ndarray) -> np.ndarray:
    """Loop through the image, and check if the row is all zeros indicating the start of the metadata bar"""

    for i, row in enumerate(img):
        if not np.any(row):
            # trim the image when the first row with all zeros is found
            break
    return img[:i]


def load_and_parse_fib_image(filename: str) -> tuple[np.ndarray, float]:
    image, pixel_size = load_image_and_metadata(filename)
    
    # from pprint import pprint
    # pprint(md)

    return image, pixel_size

def rgb_to_color_name(rgb):
    colors = {
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 0, 255): "magenta",
        (0, 255, 255): "cyan",
        (255, 255, 255): "gray",
        (0, 0, 0): "black"
    }

    # Find the color with the minimum Euclidean distance
    closest_color = min(colors.keys(), key=lambda color: sum((a-b)**2 for a, b in zip(rgb, color)))

    return colors[closest_color]


def load_and_parse_fm_image(path: str) -> Tuple[np.ndarray, dict]:
    image = tff.imread(path)

    zstep, pixel_size, colours = None, None, None
    try:
        ome = from_tiff(path)
        pixel_size = ome.images[0].pixels.physical_size_x # assume isotropic
        zstep = ome.images[0].pixels.physical_size_z

        # convert to SI (if required)
        pixel_size_unit = ome.images[0].pixels.physical_size_x_unit
        zstep_unit = ome.images[0].pixels.physical_size_z_unit
        _unit_map = {
            UnitsLength.NANOMETER: 1e-9,
            UnitsLength.MICROMETER: 1e-6,
            UnitsLength.MILLIMETER: 1e-3,
            UnitsLength.METER: 1,
        }
        pixel_size *= _unit_map[pixel_size_unit]
        zstep *= _unit_map[zstep_unit]

        colours = [channel.color.as_rgb_tuple() for channel in ome.images[0].pixels.channels]
    except Exception as e:
        logging.debug(f"Failed to extract metadata: {e}")

    if colours is None:
        colours = [(255, 255, 255) for _ in range(image.shape[0])]

    colours = [rgb_to_color_name(colour) for colour in colours]

    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    return image, {"pixel_size": pixel_size, 
                   "zstep": zstep, 
                   "colours": colours}


##### CORRELATION RESULTS #####

# convert 2D image coordinates to microscope image coordinates
def convert_poi_to_microscope_coordinates(
    poi_coordinates: np.ndarray, fib_image_shape: tuple, pixel_size_um: float
) -> list:
    # image centre
    cx = float(fib_image_shape[1] * 0.5)
    cy = float(fib_image_shape[0] * 0.5)

    poi_image_coordinates: list = []

    for i in range(poi_coordinates.shape[1]):
        px = poi_coordinates[:, i]  # (x, y, z) in pixel coordinates
        px = [float(px[0]), float(px[1])]
        px_x, px_y = (
            px[0] - cx,
            cy - px[1],
        )  # point in microscope image coordinates (px)
        pt_um = (
            px_x * pixel_size_um,
            px_y * pixel_size_um,
        )  # point in microscope image coordinates (um)
        poi_image_coordinates.append(
            {"image_px": px, "px": [px_x, px_y], "px_um": [pt_um[0], pt_um[1]]}
        )

    return poi_image_coordinates


def extract_transformation_data(transf, mod_translation, reproj_3d, delta_2d) -> dict:
    # extract eulers in degrees
    eulers = transf.extract_euler(r=transf.q, mode="x", ret="one")
    eulers = eulers * 180 / np.pi

    # RMS error
    rms_error = transf.rmsError

    # difference between points after transforming 3D points to 2D
    delta_2d_mean_abs_err = np.absolute(delta_2d).mean(axis=1)

    transformation_data = {
        "transformation": {
            "scale": float(transf.s_scalar),
            "rotation_eulers": eulers.tolist(),
            "rotation_quaternion": transf.q.tolist(),
            "translation_around_rotation_center_custom": mod_translation.tolist(),
            "translation_around_rotation_center_zero": transf.d.tolist(),
        },
        "error": {
            "reprojected_3d": reproj_3d.tolist(),
            "delta_2d": delta_2d.tolist(),
            "mean_absolute_error": delta_2d_mean_abs_err.tolist(),
            "rms_error": float(rms_error),
        },
    }

    return transformation_data


def parse_correlation_result(cor_ret: list, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret[2]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret[0]     # transformation matrix
    reproj_3d = cor_ret[1]  # reprojected 3D points to 2D points
    delta_2d = cor_ret[3]   # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret[5]  # translation around rotation center
    transformation_data = extract_transformation_data(transf=transf, 
                                                      mod_translation=mod_translation, 
                                                      reproj_3d=reproj_3d, 
                                                      delta_2d=delta_2d)

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data


def parse_correlation_result_v2(cor_ret: dict, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret["output"]["reprojected_2d_poi"]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret["output"]["transform"]     # transformation matrix
    reproj_3d = cor_ret["output"]["reprojected_3d_coordinates"]  # reprojected 3D points to 2D points
    delta_2d = cor_ret["output"]["reprojection_error"]   # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret["output"]["modified_translation"]  # translation around rotation center
    transformation_data = extract_transformation_data(transf=transf, 
                                                      mod_translation=mod_translation, 
                                                      reproj_3d=reproj_3d, 
                                                      delta_2d=delta_2d)

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data


def save_correlation_data(data: dict, path: str) -> None:
    correlation_data_filename = os.path.join(path, "correlation_data.yaml")
    with open(correlation_data_filename, "w") as file:
        yaml.dump(data, file)

    logging.info(f"Correlation data saved to: {correlation_data_filename}")

    return correlation_data_filename