from utils.base_support import mkdirs
from termcolor import cprint # 標準出力に色を付ける
from bs4 import BeautifulSoup
import os
import sys
import urllib.request
import requests
import random
import json
import time

'''
scraping setup global variables
'''
ENCODE_LIST = [
        'iso-2022-jp',
        'iso-2022-jp-1',
        'iso-2022-jp-2',
        'iso-2022-jp-3',
        'iso-2022-jp-ext',
        'iso-2022-jp-2004',
        'euc-jisx0213',
        'euc-jis-2004',
        'enc-jp',
        'utf-7',
        'utf-8',
        'utf-16',
        'utf-16-be',
        'utf-16-le',
        'cp932', # 日本語が文字化けしない
        'shift-jis', # 日本語が文字化けしない
        'shift-jisx0213', # 日本語が文字化けしない
        'shift-jis-2004' # 日本語が文字化けしない
        ]

OSNAME = os.name

if OSNAME == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

'''
Image-scraping functions and classes.
'''
class ImageScraping:
    def __init__(self, site='google', seed=None, savedir=None):
        if site == 'google':
            self.baseurl = 'https://www.google.co.jp/search'
        elif site == 'yahoo':
            self.baseurl = 'https://search.yahoo.co.jp/image/search'
        elif site == 'bing':
            self.baseurl = 'https://www.bing.com/images/search'

        # sessionを使うと一連のHTTPリクエスト・レスポンスでCookieやカスタムヘッダーを使いまわすことができる
        self.sess = requests.session()
        # セッションにおけるユーザーエージェントの変更(偽装)
        # この部分によってレスポンスが異なったりすることがある
        if seed is not None:
            if (type(seed) is int) and (seed in range(10000)):
                random.seed(seed)
                nrand1 = float(random.randint(0,10000))
                nrand2 = float(random.randint(0,10000))
            else:
                raise TypeError('seed: int, range of 0-9999')
        elif seed is None:
            nrand1 = float(random.randint(0,10000))
            nrand2 = float(random.randint(0,10000))

        self.sess.headers.update(
                {
                    'User-Agent':'Mozilla/5.0 (X11; Linux x86_64; rv{:.1}) \
                            Gecko/20100101 Firefox/{:.1}'.format(nrand1, nrand2)
                    }
                )

        self.savedir = mkdirs(savedir)

    def run_scrape(self, keyword='', n_img=10):
        if type(keyword) is str:
            query = self.__gen_query(keyword)

            results = self.__search_imgs(query, n_img)

            err_counter = 0
            for i, url in enumerate(results):
                bf = os.path.basename(url)
                dlpath = os.path.join(self.savedir,bf)

                sys.stdout.write('Downloading: [{}]'.format(bf)+NEWLINECODE)

                try:
                    urllib.request.urlretrieve(url, dlpath) # ファイルをダウンロードするときはurlretrieveを使う
                    cprint('Successful.', 'green', attrs=['bold'])
                except urllib.error.HTTPError as err:
                    cprint('Error: {}'.format(str(err)), 'yellow', attrs=['bold'])
                    err_counter += 1
                    continue
                except urllib.error.URLError as err:
                    cprint('Error: {}'.format(str(err)), 'yellow', attrs=['bold'])
                    err_counter += 1
                    continue
                except UnicodeEncodeError as err:
                    cprint('Error: {}'.format(str(err)), 'yellow', attrs=['bold'])
                    err_counter += 1
                    continue

            cprint('Download acomplished.', 'cyan', attrs=['bold'])
            cprint('Successful DL: {}'.format(len(results)-err_counter), 'cyan', attrs=['bold'])

            if err_counter:
                cprint('Falied DL: {}'.format(err_counter), 'red', attrs=['bold'])
        else:
            raise TypeError('keyword: str')

    def __gen_query(self, keyword):
        if '@' in keyword:
            keyword = keyword.replace('@',' ')

        pages = 0
        while True:
            # 第一引数に辞書を与えると, クエリ文字列を作成
            # ちなみにリストでも可能, その時は(key,value)でリスト化する

            # keywordをspace区切りで色々にすると+でクエリパラメータにしてくれるのかな？
            if 'google' in self.baseurl:
                q_str = urllib.parse.urlencode(
                        {
                            'q':keyword,
                            'tbm':'isch', # 検索の種類・・・画像検索, 動画ならvid
                            'ijn':str(pages) # ひょっとしてページ番号?, スクロールを考慮してるのかも
                            }
                        )
                yield self.baseurl+'?'+q_str
            # 以下については今後変更予定
            elif 'yahoo' in self.baseurl:
                q_str = urllib.parse.urlencode(
                        {
                            'p':keyword,
                            'ei':'UTF-8'
                            }
                        )
                yield self.baseurl+';'+q_str
            elif 'bing' in self.baseurl:
                q_str = urllib.parse.urlencode(
                        {
                            'q':keyword,
                            'tbm':'isch'
                            #'ijn':str(pages)
                            }
                        )
                yield self.baseurl+'?'+q_str

            pages += 1

    def __search_imgs(self, gen_query, n_img=10):
        results = list() # 見つけたデータのURLを格納するリスト
        total = 0 # 見つけた画像データのカウンター

        while True:
            # クエリ文字列のジェネレーターを再帰的に参照して, リクエストを取得
            html = self.sess.get(next(gen_query)).text

            try:
                soup = BeautifulSoup(html, 'lxml')
            except ImportError as err:
                cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
                sys.exit(1)

            # BeautifulSoupのselectメソッドでcssセレクターでタグを取得可能
            elements = soup.select('.rg_meta.notranslate')
            # 得たcss属性要素(json)をdictに変換
            jsons = [json.loads(e.get_text()) for e in elements]
            image_url_list = [js['ou'] for js in jsons]

            if not image_url_list:
                cprint('No more images..', 'red', attrs=['bold'])
                break
            elif len(image_url_list)>n_img-total:
                results += image_url_list[:n_img-total]
                break
            else:
                results += image_url_list
                total += len(image_url_list)

            self.__scrape_sleep()

        cprint('Found Image Data: {}.'.format(len(results)), 'green', attrs=['bold'])

        return results

    def __scrape_sleep(self):
        time1 = 402
        time2 = 238
        time3 = 45
        epsilon1 = random.uniform(4,10)
        epsilon2 = random.uniform(1,3)
        epsilon3 = random.uniform(0,2)
        delta = random.randint(0,3)

        sleep_time = time1*epsilon1+time2*epsilon2+time3*epsilon3-delta

        time.sleep(sleep_time)
