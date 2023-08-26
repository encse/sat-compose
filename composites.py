import os
from PIL import Image
import math
import argparse
import numpy as np

def rotate(image):
    pixels = np.array(image)
    pixels = np.fliplr(pixels)
    pixels = np.flipud(pixels)
    return Image.fromarray(pixels)

def make_composite(image_r, image_g, image_b):
    pixels = np.array(image_r)
    pixels[:,:,1] = np.array(image_g)[:,:,0]
    pixels[:,:,2] = np.array(image_b)[:,:,0]

    return Image.fromarray(pixels)

def equalize(image: Image):
    pixels = np.array(image)

    for channel in range(3):
        histogram, _ = np.histogram(pixels[:,:,channel], bins=256, range=(0, 256))
        scaling = histogram.cumsum()  * 255 / (image.width * image.height)

        # map the pixels of the channel based on the scaling this is similar to doing this but vectorized
        #     for row in rows:
        #         for column in columns:
        #             pixels[row, column][0] = scaling[pixels[row, column][0]]

        pixels[:,:,channel] = scaling[pixels[:,:,channel]]

    pixels = pixels.clip(0, 255)
    pixels = pixels.astype(np.uint8)
    return Image.fromarray(pixels)

# This was mostly based off the following document :
# https://web.archive.org/web/20200110090856if_/http://ceeserver.cee.cornell.edu:80/wdp2/cee6150/Monograph/615_04_GeomCorrect_rev01.pdf
def correct_earth_curvature(image: Image):
    pixels = np.array(image)

    EARTH_RADIUS = 6371
    swath = 2800
    resolution_km = 1
    satellite_height = 820

    # Compute the satellite's orbit radius
    satellite_orbit_radius = EARTH_RADIUS + satellite_height
    # Compute the output image size, or number of samples from the imager
    corrected_width = round(swath / resolution_km)
    # Compute the satellite's view angle 
    satellite_view_angle = swath / EARTH_RADIUS 
    #  Max angle relative to the satellite
    edge_angle = -math.atan(EARTH_RADIUS * math.sin(satellite_view_angle / 2) / ((math.cos(satellite_view_angle / 2)) * EARTH_RADIUS - satellite_orbit_radius))

    # Create a LUT to avoid recomputing on each row
    correction_factors = np.full(corrected_width, 0, dtype=np.uint32)
    for i in range(corrected_width):
        # Get the satellite's angle
        angle = ((i / float(corrected_width)) - 0.5) * satellite_view_angle
        # Convert to an angle relative to earth
        satellite_angle = -math.atan(EARTH_RADIUS * math.sin(angle) / ((math.cos(angle)) * EARTH_RADIUS - satellite_orbit_radius))
        # Convert that to a pixel from the original image
        correction_factors[i] = (image.width - 1) * ((satellite_angle / edge_angle + 1.0) / 2.0)

    indices = np.arange(corrected_width) # 0,1,2....
    pixels = pixels[:, correction_factors[indices]]
    return Image.fromarray(pixels)

# white balance algorithm from gimp
def white_balance(image: Image, percentileValue):
    pixels = np.array(image)

    percentile1 = np.percentile(pixels, percentileValue, axis=(0,1))
    percentile2 = np.percentile(pixels, 100 - percentileValue, axis=(0,1))

    pixels = (pixels - percentile1) * 255 / (percentile2 - percentile1)

    pixels = pixels.clip(0, 255)
    pixels = pixels.astype(np.uint8)
    return Image.fromarray(pixels)

def linear_invert(image: Image):
    pixels = np.array(image)
    pixels = 255 - pixels
    return Image.fromarray(pixels)

def normalize(image: Image):
    pixels = np.array(image)

    min_rgb = np.min(pixels, axis=(0, 1))
    max_rgb = np.max(pixels, axis=(0, 1))

    # Avoid division by 0
    if np.any(max_rgb == min_rgb):
        return image 

    # Compute scaling factor
    scale = 255 / (max_rgb - min_rgb)

    # Scale entire image
    pixels = (pixels - min_rgb) * scale
    
    pixels = pixels.clip(0, 255)
    pixels = pixels.astype(np.uint8)
    return Image.fromarray(pixels)

def run_all(path):
    layer1 = path + '_64.bmp'
    layer2 = path + '_65.bmp'
    layer3 = path + '_66.bmp'

    if os.path.exists(layer1) and os.path.exists(layer2):

        print("making composite")
        image = make_composite(Image.open(layer2), Image.open(layer2), Image.open(layer1))
        image.save(f"{path}_221.jpg")

        print("correct Earth curvature")
        image = correct_earth_curvature(image)
        image.save(f"{path}_corrected.jpg")

        print("equalize")
        image = equalize(image)
        image.save(f"{path}_equalized.jpg")

        print("white balance")
        image = white_balance(image, 0.05)
        image.save(f"{path}_white_balanced.jpg")
        
        print("normalize")
        image = normalize(image)
        image.save(f"{path}_normalized.jpg")

        print("invert")
        image = linear_invert(image)
        image.save(f"{path}_inverted.jpg")
        
        print("rotate")
        image = rotate(image)
        image.save(f"{path}_rotated.jpg")
       

def main():
    parser = argparse.ArgumentParser(description="A script that takes one parameter from the command line")
    parser.add_argument("input", help="The parameter you want to process")
    
    args = parser.parse_args()
    if not args.input:
        parser.print_help()
    else:
        run_all(args.input)

if __name__ == "__main__":
    main()
