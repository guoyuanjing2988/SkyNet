import random
import sys
import math

def generate_radial(num_scraper, lat, lng, radius=10000):

    radius = radius / 111300
    coordinates = []
    for _ in range(num_scraper):
        u = float(random.uniform(0.0, 1.0))
        v = float(random.uniform(0.0, 1.0))

        w = radius * math.sqrt(u)
        t = 2 * math.pi * v
        x = w * math.cos(t)
        y = w * math.sin(t)

        random_lat = x + lat
        random_lng = y + lng

        coordinates.append([random_lat, random_lng])
    return coordinates
