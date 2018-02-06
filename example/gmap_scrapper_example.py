import sys
sys.path.append('../')

import multiprocessing
import logging
import os
from time import time
from queue import Queue
from skynet.scraper.gmap_scraper import GoogleMapScraper
from skynet.scraper.gmap_thread import GmapWorker

DIRECTORY = '../data'
NUM_WORKER = multiprocessing.cpu_count()

if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

ts = time()

gms = [GoogleMapScraper(1.3403, 103.9629, DIRECTORY),
       GoogleMapScraper(25.204849, 55.270783, DIRECTORY),
       GoogleMapScraper(20.593684, 78.96288, DIRECTORY),
       GoogleMapScraper(20.617100, 78.931897, DIRECTORY),
       GoogleMapScraper(12.839045, 30.203251, DIRECTORY)]

queue = Queue()

for i in range(NUM_WORKER):
    worker = GmapWorker(queue)
    worker.daemon = True
    worker.start()

for scraper in gms:
    queue.put(scraper)

queue.join()
print("Time taken: {}".format(time() - ts))
