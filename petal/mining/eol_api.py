import requests, argparse, json, sys
from pprint import pprint
from time import sleep, time

class EOL_API:
    def __init__(self):
        self.url = 'https://eol.org/service/cypher'
        with open('api.token', 'r') as infile:
            api_token = infile.read().strip()
        self.headers = {"accept": "application/json",
                        "authorization": "JWT " + api_token}

        self.format = 'cypher'

    def search(self, query, filter_data=True):
        url = self.url
        data = {"query": query, "format": self.format}
        r = requests.get(url,
                         stream=(format=="csv"),
                         headers=self.headers,
                         params=data)
        if r.status_code != 200:
            sys.stderr.write('HTTP status %s\n' % r.status_code)
        ct = r.headers.get("Content-Type").split(';')[0]
        if ct == "application/json":
            j = {}
            try:
                j = r.json()
                return j
            except ValueError:
                sys.stderr.write('JSON syntax error\n')
                print(r.text[0:1000], file=sys.stderr)
                sys.exit(1)
        else:
            sys.stderr.write('Unrecognized response content-type: %s\n' % ct)
            print(r.text[0:10000], file=sys.stderr)
            sys.exit(1)
        if r.status_code != 200:
            sys.exit(1)

    def page(self, finder, query, page_size=100, rate_limit=.25):
        count_query = finder + ' WITH COUNT (n) AS count RETURN count LIMIT 1'
        count       = self.search(count_query)
        count       = count['data'][0][0]
        print('Paging over {} items with page size {} and {} extra seconds between pages'.format(count, page_size, rate_limit))

        for i in range(count // page_size):
            skip = i * page_size
            page_query = query + ' SKIP {skip} LIMIT {limit}'.format(skip=skip, limit=page_size)
            yield self.search(page_query)
            sleep(rate_limit)


if __name__ == '__main__':
    api = EOL_API()
    species = 1.9e6
    total = 0
    start = time()
    for item in api.page('MATCH (n:Page) WHERE n.rank = \'species\'', 'MATCH (species:Page) WHERE species.rank = \'species\' RETURN species.canonical'):
        total += (len(item['data']))
        duration = time() - start
        species_per_sec = total / duration
        total_seconds  = species / species_per_sec
        eta_seconds = total_seconds - duration
        eta = eta_seconds / 3600
        percent = duration / total_seconds
        pprint(item)
        1/0 
        print('Species: {}, Rate: {} species per second, ETA: {}h, Percent: {}\r'.format(total, round(species_per_sec, 1), round(eta, 1), round(percent, 5)), flush=True, end='')
    print(total)
