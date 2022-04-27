from queue import Queue, Empty
from pathlib import Path
import re
from threading import Thread, Event
import numpy as np

def ensure_folders(path):
    match = re.match("(.*)/.*?$",path)
    if match:
        Path(match[1]).mkdir(parents=True, exist_ok=True)

def numpy_writer(path,val):
    def lam():
        ensure_folders(path)
        np.save(path,val)
    return lam

class KillSignal:
    def __init__(self):
        pass

def process(queue):
    while True:
        lam = queue.get()
        if isinstance(lam,KillSignal):
            break
        if lam is None:
            import pdb; pdb.set_trace()
        lam()



class AsyncLambdaRunner:
    def __init__(self,threads = 1):
        self.queue = Queue()
        self.threads = [Thread(target=process,args=(self.queue,)) for _ in range(threads)]

    def put(self,item):
        self.queue.put(item)

    def start(self):
        for t in self.threads:
            t.start()
    def join(self):
        for _ in self.threads:
            self.put(KillSignal())
        for t in self.threads:
            t.join()
        # while True:
            # lam = self.queue.get()
            # lam()
            # match = re.match("(.*)/.*?$",path)
            # if match:
                # Path(mach[1]).mkdir(parents=True, exist_ok=True)
            # np.savelllllll
            # import pdb; pdb.set_trace()
            # path_prefix = path.
            # Path("/my/directory").mkdir(parents=True, exist_ok=True)
            # ll

if __name__ == '__main__':
    writer = AsyncLambdaRunner()
    writer.put(lambda: print("test"))
    writer.put(lambda: print("test2"))
    writer.put(lambda: print("test3"))
    writer.start()
    writer.join()
