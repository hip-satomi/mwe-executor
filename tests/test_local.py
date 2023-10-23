import unittest
import os
from git_utils import get_git_url
import mlflow
import json
import requests
from io import BytesIO
from PIL import Image
import json
from pathlib import Path

from git_utils import get_git_revision_short_hash


class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # download the image
        import requests

        url = 'https://fz-juelich.sciebo.de/s/wAXbC0MoN1G3ST7/download'
        r = requests.get(url, allow_redirects=True)

        print(len(r.content))

        with open('test.png', 'wb') as file:
            file.write(r.content)

    def test_prediction(self, entrypoint="main"):
        """Test segmentation executor on a single image in local execution
        """
        run = mlflow.projects.run(
            './',
            entry_point=entrypoint,
            backend="local",
            parameters={
                "input_images": "test.png"
            }
        )

        # download the output artifact
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(run.run_id, 'output.json', './')

        # make sure the output file exists
        self.assertTrue(Path('output.json').exists())

        with open('output.json', 'r') as input_file:
            # load json data
            data = json.load(input_file)

            # we expect a single frame
            self.assertEqual(len(data["segmentation_data"]), 1)

            # we expect a single detection
            self.assertEqual(len(data["segmentation_data"][0]), 1)

            # with 4 contour points
            self.assertEqual(len(data["segmentation_data"][0][0]["contour_coordinates"]), 4)

    def test_info_entrypoint(self):
        """Test the info entrypoint of the executor
        """
        run = mlflow.projects.run(
            './',
            entry_point="info",
            backend='local',
        )

        # download the output artifact
        client = mlflow.tracking.MlflowClient()
        client.download_artifacts(run.run_id, 'output.json', './')

        with open('output.json', 'r') as input_file:
            info_result = json.load(input_file)
            self.assertTrue(info_result['name'] == 'mwe-executor')
            self.assertTrue(info_result['git_hash'] == get_git_revision_short_hash())
            self.assertTrue(info_result["git_url"] == get_git_url())
            self.assertTrue(info_result["type"] == "info")


if __name__ == '__main__':
    unittest.main()
