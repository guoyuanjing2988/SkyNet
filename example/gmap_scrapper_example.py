import sys
sys.path.append('../')

import multiprocessing
import logging
import numpy as np
import os
from time import time
from queue import Queue
from skynet.scraper.gmap_generator import generate_radial
from skynet.scraper.gmap_scraper import GoogleMapScraper
from skynet.scraper.gmap_thread import GmapWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(threadName)s %(message)s')

DIRECTORY = '../data'
NUM_SCRAPER = 10
NUM_WORKER = multiprocessing.cpu_count()

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

ts = time()

# Singapore
sg_coordinates = generate_radial(NUM_SCRAPER, 1.290270, 103.851959, 1000)

# US New York
ny_coordinates = generate_radial(NUM_SCRAPER, 40.730610, -73.935242, 1000)

# US Kentucky
ky_coordinates = generate_radial(NUM_SCRAPER, 38.047989, -84.501640, 1000)

# Brazil Sao Paulo
br_coordinates = generate_radial(NUM_SCRAPER, -23.533773, -46.625290, 1000)

places = [sg_coordinates, ny_coordinates, ky_coordinates, br_coordinates]

queue = Queue()

for i in range(NUM_WORKER):
    worker = GmapWorker(queue)
    worker.daemon = True
    worker.start()

for coordinates in places:
    for index, (lat, lng) in enumerate(coordinates):
        logger.info('Queueing %s GoogleMapScraper out of %s', index, NUM_SCRAPER)
        queue.put(GoogleMapScraper(lat, lng, DIRECTORY))

queue.join()

print("Time taken: {}".format(time() - ts))
