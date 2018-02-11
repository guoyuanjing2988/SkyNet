from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

class GoogleMapScraper:

    def __init__(self, lat, lng, directory, width=300, height=300, zoom=18, api_key='AIzaSyDP3-YvP9o7xTc31H0NfOSXC7cXtaNFatk'):
        self._lat = lat
        self._lng = lng
        self._width = width
        self._height = height
        self._zoom = zoom
        self._api_key = api_key
        self._directory = directory
        self._file_name = str(self._lat) + "_" + str(self._lng) + ".png"

    def generate_image(self):
        logging.basicConfig(level=logging.DEBUG, format='%(threadName)s %(message)s')
        request = 'https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=' + str(self._lat) + ',' + str(self._lng) + '&zoom=' + str(self._zoom) + '&size=' + str(self._width) + 'x' + str(self._height) + '&key=' + str(self._api_key)

        full_path = os.path.join(self._directory, self._file_name)
        logger.info('Downloading file: %s to directory: %s', self._file_name, self._directory)

        try:
            urllib.request.urlretrieve(request, full_path)
        except IOError:
            raise Exception("Could not generate the image - try adjusting the zoom level and checking your coordinates")
        else:
            logger.info("The image %s has successfully been saved", self._file_name)
