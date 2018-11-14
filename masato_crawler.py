import os
import sys
import time
import bs4
import urllib.request
import argparse
import subprocess
import random

def main():
    url, ext_list = arguments()

    crawler = ImageCrawler(url, ext_list)
    crawler()

def arguments():
    p = argparse.ArgumentParser()
    p.add_argument(
            "-u","--url",help="the URL of wanted images",
            required=True
            )
    p.add_argument(
            "-e","--extensions",help="the extensions of images",
            required=False, default=".jpg,.png,.bmp,.gif,.jpeg"
            )
    p_args = p.parse_args()

    url = p_args.url
    extensions = p_args.extensions.split(",")

    return url, extensions

def make_directories(dir_path):
    if not os.path.exists(dir_path):
        command = "mkdir -p {0}".format(dir_path)
        return_code = subprocess.call(command.split())
        assert return_code == 0,\
                "\nNot create this directiories...[{0}]\n".format(dir_path)
    return dir_path

class ImageCrawler(object):
    def __init__(self, url, extensions):
        self.save_dir = make_directories("./imgs_save")
        self.url = url
        self.extensions = extensions

    def __call__(self):
        self.html = self.get_html_string()
        assert len(self.html) >= 1,\
                "\nNot get HTML from this URL...\n"
        self.get_resource()

    def get_html_string(self):
        decoded_html = ""
        try:
            request = urllib.request.urlopen(self.url)
            html = request.read()
        except:
            return decoded_html

        enc = self.check_encoding(html)
        if enc == None:
            return decoded_html

        decoded_html = html.decode(enc)
        return decoded_html

    def get_resource(self):
        resource_list = []
        soup = bs4.BeautifulSoup(self.html, "lxml")
        for a_tag in soup.find_all("a"):
            href_str = a_tag.get("href")
            try:
                path, ext = os.path.splitext(href_str)
                if ext in self.extensions:
                    resource_list.append(href_str)
            except:
                pass

        resource_list = sorted(set(resource_list), key=resource_list.index)
        for resource in resource_list:
            img_name = os.path.basename(resource)
            try:
                sys.stdout.write("download .... [{0}]".format(img_name))
                request = urllib.request.urlopen(resource)
                with open(os.path.join(self.save_dir, img_name), "wb") as save_f:
                    save_f.write(request.read())
            except Exception as err:
                sys.stdout.write(err)
                sys.stdout.write("download failed .... [{0}]".format(img_name))
            finally:
                time_list = [i for i in range(1,101)]
                time.sleep(random.choice(time_list))

    def check_encoding(self, byte_str):
        encoding_list = [
                "utf-8", "utf_8", "enc_jp",
                "euc_jis_2004", "euc_jisx_0213", "shift_jis",
                "shift_jis_2004", "shift_jisx0213", "iso2022jp",
                "iso2022_jp_1", "iso2022_jp_2", "iso2022_jp_3",
                "iso2022_jp_ext", "latin_1", "ascii"
                ]
        for enc in encoding_list:
            try:
                byte_str.decode(enc)
                break
            except:
                enc = None
        return enc

if __name__ == "__main__":
    main()
