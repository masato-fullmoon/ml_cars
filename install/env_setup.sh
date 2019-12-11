#!/bin/bash

# ---------- functions ----------

function existscommand() {
    # コマンドの存在はこれで確認できる
    type -a $1 > /dev/null 2>&1
}

function anaconda_install() {
    if [ -e `python --version` | grep 'Python 2.' ]; then
        echo 'This Python is 2.x.x, you should install Python 3.x.x'

        SAVEBASEDIR=$1 # 保存先ディレクトリ, ホームディレクトリが楽
        INSTALLERURL=`python anaconda3_installer_py2.py $2`
    elif [ -e `python --version` | grep 'Python 3.' ]; then
        SAVEBASEDIR=$1
        INSTALLERURL=`python anaconda3_installer_py3.py $2`
    fi

    if ! existscommand wget; then
        # curlコマンドで取得したファイル名を指定したディレクトリに送る方法
        cd ${SAVEBASEDIR} && { curl -O ${INSTALLERURL} ; cd -; }
    else
        wget -P ${SAVEBASEDIR} ${INSTALLERURL}
    fi

    echo 'Anaconda3 install acomplished.'
}

# ---------- main programs -----------

if [ `uname -s` == 'Darwin' ]; then
    OS='MacOS' # Macにインストールする人
else
    OSTYPENAME=`expr substr $(uname -s) 1 5`

    if [ ${OSTYPENAME} == 'MINGW' ]; then
        OS='Windows' # Git-for-Windowsとかにインストールする人
    elif [ ${OSTYPENAME} == 'Linux' ]; then
        OS='Linux' # 我らがLinuxにインストールする人
    else
        echo 'Unknown os type: Darwin/MINGW/Linux'
        exit 1;
    fi
fi

if ! existscommand conda; then
    echo 'conda is not found.'

    PROCESSFLAG=0

    while [ ${PROCESSFLAG} -eq 0 ]
    do
        echo 'auto-install anaconda3 installer: [yes/no] >>> '
        read INSTALLFLAG

        if [ ${INSTALLFLAG} == 'yes' ]; then
            echo 'Start to install Anaconda3.'
            anaconda_install $HOME ${OS}
            break
        elif [ ${INSTALLFLAG} == 'no' ]; then
            echo 'Stop installing anaconda3, try again.'
            break
        else
            continue
        fi
    done
else
    echo 'Anaconda existed, please check whether or not anaconda3.'
fi
