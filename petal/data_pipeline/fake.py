import requests
url = 'https://httpbin.org/user-agent'
url = 'https://eol.org/search?utf8=%E2%9C%93&q=Drosophila'
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
headers = {'User-Agent': user_agent}
response = requests.get(url, headers=headers)
html = response.content
print(response.content)
