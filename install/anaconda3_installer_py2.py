from bs4 import BeautifulSoup
import urllib2
import sys

ANACONDA_BASEURLPATH = 'https://www.anaconda.com/distribution/'
OSTYPE = sys.argv[1]

def get_requests(url=None):
    if (url is not None) and (type(url) is str):
        html = urllib2.urlopen(url)
    else:
        raise TypeError('url: string URL path-like')

    try:
        soup = BeautifulSoup(html, 'lxml')
    except Exception as err:
        print 'Error: {}'.format(str(err))
        print 'Install lxml-parser, pip install lxml'
        sys.exit(1)

    return soup

if __name__ == '__main__':
    base_soup = get_requests(ANACONDA_BASEURLPATH)

    hrefs = list(set(base_soup.find_all('a', class_='more')))

    for h in hrefs:
        repos_url = h.get('href')

        if (OSTYPE in repos_url) and ('Anaconda3' in repos_url):
            print repos_url
            break
