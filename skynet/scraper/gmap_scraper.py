import urllib.request
import os

class GoogleMapScraper:

    def __init__(self, lat, lng, width=640, height=400, zoom=18, api_key='AIzaSyD7atkiOxRi2B0nqg2VOkcFmgOefC_N7TU', directory='data/'):
        self._lat = lat
        self._lng = lng
        self._width = width
        self._height = height
        self._zoom = zoom
        self._api_key = api_key
        self._directory = directory

    def generate_image(self):
        request = 'https://maps.googleapis.com/maps/api/staticmap?maptype=satellite&center=' + str(self._lat) + ',' + str(self._lng) + '&zoom=' + str(self._zoom) + '&size=' + str(self._width) + 'x' + str(self._height) + '&key=' + str(self._api_key)

        if not os.path.exists(self._directory):
            os.makedirs(self._directory)

        full_path = os.path.join(self._directory, str(self._lat) + "_" + str(self._lng) + ".png")

        try:
            urllib.request.urlretrieve(request, full_path)
        except IOError:
            print("Could not generate the image - try adjusting the zoom level and checking your coordinates")
        else:
            print("The image " + str(self._lat) + ", " + str(self._lng) + " has successfully been saved")
