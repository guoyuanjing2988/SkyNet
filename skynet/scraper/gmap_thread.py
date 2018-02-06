from threading import Thread

class GmapWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            gmap_object = self.queue.get()
            gmap_object.generate_image()
            self.queue.task_done()
