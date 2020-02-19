from multiprocessing import Process
from time import sleep
import wikipedia

def test():
    x = 0
    while True:
        page = wikipedia.page(wikipedia.random()[0])
    return x

def main():
    pool = []
    for i in range(10):
        pool.append(Process(target=test, args=()))
    for p in pool:
        p.start()

if __name__ == '__main__':
    wikipedia.set_rate_limiting(False)
    main()
