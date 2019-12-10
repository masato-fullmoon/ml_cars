import sys
import os
import argparse
import urllib

if os.name == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

PYTHON_VERSION = sys.version_info[0]

if PYTHON_VERSION == 2:
    from urlparse import urljoin
elif PYTHON_VERSION == 3:
    from urllib.parse import urljoin
    import requests

try:
    from bs4 import BeautifulSoup
except ImportError as err:
    sys.stdout.write('Error: {}'.format(str(err))+NEWLINECODE)
    sys.exit(1)

def cmd_args():
    p = argparse.ArgumentParser()

    p.add_argument('--basepath', help='anaconda distribution path',
            type=str, default='https://www.anaconda.com/distribution/', required=False)
    p.add_argument('--ostype', help='os type to install anaconda3',
            type=str, required=True)

    return p.parse_args()

def get_requests(url=None):
    if (url is not None) and (type(url) is str):
        if PYTHON_VERSION == 2:
            reqdata = urllib.urlopen(url).read()
        elif PYTHON_VERSION == 3:
            reqdata = requests.get(url)
    else:
        raise TypeError('url: string URL path-like')

    try:
        soup = BeautifulSoup(reqdata.content, 'lxml')
    except ImportError as err:
        sys.stdout.write('Error: {}'.format(str(err))+NEWLINECODE)
        sys.stdout.write('Install lxml-parser, pip install lxml'+NEWLINECODE)
        sys.exit(1)

    return soup

if __name__ == '__main__':
    commandlineargs = cmd_args()

    baseurl = commandlineargs.basepath
    ostype = commandlineargs.ostype

    base_soup = get_requests(baseurl)

    hrefs = list(set(base_soup.find_all('a', class_='more')))

    for h in hrefs:
        repos_url = h.get('href')

        if (ostype in repos_url) and ('Anaconda3' in repos_url):
            print(repos_url)
            break
