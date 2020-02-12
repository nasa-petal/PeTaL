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
