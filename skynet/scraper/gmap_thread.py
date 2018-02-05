import threading
from gmap_scraper import GoogleMapScraper

def create_thread(scraper_object):
    dl_thread = threading.Thread(target=scraper_object.generate_image)
    dl_thread.start()

if __name__ == '__main__':

    gms = [GoogleMapScraper(1.3403, 103.9629),
           GoogleMapScraper(25.204849, 55.270783),
           GoogleMapScraper(20.593684, 78.96288),
           GoogleMapScraper(20.617100, 78.931897),
           GoogleMapScraper(12.839045, 30.203251)]

    for i in range(len(gms)):
        create_thread(gms[i])
