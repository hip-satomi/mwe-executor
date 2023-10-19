import numpy as np
import os
import sys
import json
from PIL import Image
import glob
import mlflow


from git_utils import get_git_revision_short_hash, get_git_url


import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('images', type=str, nargs='+',
                        help='list of images')
    parser.add_argument('--omni', action="store_true", help="Use the omnipose model")

    args = parser.parse_args()

    # Versioning: get the git hash of the current commit
    short_hash = get_git_revision_short_hash()
    try:
        git_url = get_git_url()
    except:
        # repo might be under development and not have a url
        git_url = "local"


    # 1. load all the images for segmentation
    if len(args.images) == 1:
        image_path = args.images[0]
        if os.path.isdir(image_path):
            # it's a folder, iterate all images in the folder
            args.images = sorted(glob.glob(os.path.join(image_path, '*.png')))
        else:
            # it may be a list of images
            args.images = image_path.split(' ')

    images = [np.asarray(Image.open(image_path)) for image_path in args.images]

    # 2. Perform segmentation

    segmentation_results = []

    # loop over all images in order to segment them
    for image in images:

        # TODO: here you should apply the segmentation onto the image and convert the result into contour points for the individual instances
        # we use one dummy contour as an example here
        contour_list = [
            [[50, 50], [300, 50], [300, 300], [50, 300]]
        ]

        segmentation = [dict(
            label = 'Cell',                 # indicates the label of the recognized object
            contour_coordinates = contour,  # contour coordinates of the object
            type = 'Polygon'                # indicates the type of object. for now, only polygons are supported
        ) for contour in contour_list]

        # add this to the result
        segmentation_results.append(segmentation)

    # 3. Save results: package everything into a dictionary ready for serialization
    result = dict(
        model = f'{git_url}#{short_hash}',
        format_version = '0.2',
        segmentation_data = segmentation_results
    )

    # write to json file
    with open('output.json', 'w') as output:
        json.dump(result, output)

    # log with mlflow (this allows segServe to obtain the results lateron)
    mlflow.log_artifact('output.json')
