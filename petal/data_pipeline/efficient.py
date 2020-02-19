import wikipedia
from multiprocessing import Process

def downloader(item):
    while True:
        page = wikipedia.page(item)
    # print(page.title, flush=True)

def main():
    processes = []
    for i in range(10):
        processes.append(Process(target=downloader, args=('Engineering',)))
    for process in processes:
        process.start()

if __name__ == '__main__':
    main()
