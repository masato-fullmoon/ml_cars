from termcolor import cprint
import multiprocessing
import os
import sys
import glob
import time
import subprocess
import slackweb
import requests
import datetime
import json

'''
utils setup global variables
'''
JSONFILE = os.path.join(os.path.expanduser('~'),'token.json')
LINE_NOTIFY_URL = 'https://notify-api.line.me/api/notify'

OSNAME = os.name

if OSNAME == 'nt':
    NEWLINECODE = '\r\n'
else:
    NEWLINECODE = '\n'

MODULES_BASEDIR = os.path.dirname(os.path.abspath(__file__))

'''
system supporting functions and classes
'''

def act_command(command=None):
    try:
        return_code = subprocess.call(command.split())
        assert return_code == 0,\
                'Not activate command: [{}]'.format(command)
    except:
        cprint('Use os.system to activate [{}]'.format(
            command), 'green', attrs=['bold'])
        os.system(command)

def mkdirs(dirpath=None, response=True):
    if (dirpath is not None) and (type(dirpath) is str):

        if not os.path.exists(dirpath):
            if OSNAME == 'nt':
                command = 'mkdir {}'.format(dirpath)
            else:
                command = 'mkdir -p {}'.format(dirpath)

            act_command(command)

        if response:
            return dirpath
    else:
        raise TypeError('dirpath: str')

def num_cpus_checker():
    # 返すのは論理コア数のみ
    return multiprocessing.cpu_count()

def num_gpus_checker():
    if OSNAME == 'posix':
        scriptpath = os.path.join(MODULES_BASEDIR, 'linux_nvidia_gpuschecker.sh')
        gpuspath = os.path.join(MODULES_BASEDIR, 'gpus.log')

        act_command('bash {}'.format(scriptpath))

        if os.path.exists(gpuspath):
            with open(gpuspath, 'r') as g:
                num_gpus = len(g.readlines())

            os.remove(gpuspath)

            return num_gpus
        else:
            raise FileNotFoundError('GPUs logfile is None.')
    else:
        raise OSError('Sorry, Linux only.')

class Timer:
    def __init__(self, api='line', output=True):
        if not os.path.exists(JSONFILE):
            raise FileNotFoundError('Not found your API URL json.')

        with open(JSONFILE, 'r') as t:
            tokens = json.load(t)

            if api == 'slack':
                self.token = tokens['slack_token']
                self.s = slackweb.Slack(url=self.token)
            elif api == 'line':
                self.token = tokens['line_token']
                self.s = None
            else:
                raise ValueError('API: Slack or LINE')

        self.api = api
        self.output = output

    def timer(self, func):
        def __wrapper(*args, **kwargs):
            start = time.time()
            act_now = datetime.datetime.now()

            res = func(*args, **kwargs)

            s = time.time()-start
            m = s/60
            h = m/60
            d = h//24
            msg = '''
[{}]
Function: [{}]
Act-Time:
    {:.3f} [sec]
    {:.3f} [min]
    {:.3f} [hours]
    {:.3f} [days]
            '''.format(act_now, func.__name__,
                    s,m,h,d)

            if self.api == 'slack':
                self.__send_to_slack(msg)
            elif self.api == 'line':
                self.__send_to_line(msg)

            if self.output:
                cprint(msg, 'cyan', attrs=['bold'])

            if res is not None:
                return res

        return __wrapper

    def __send_to_slack(self, msg):
        try:
            self.s.notify(text=msg)
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
            sys.exit(1)

    def __send_to_line(self, msg):
        payload = {'message':msg}
        headers = {'Authorization':'Bearer '+self.token}

        try:
            response = requests.post(
                    LINE_NOTIFY_URL, data=payload, headers=headers
                    )
        except Exception as err:
            cprint('Error: {}'.format(str(err)), 'red', attrs=['bold'])
            sys.exit(1)
