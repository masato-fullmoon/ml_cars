#!/bin/bash

# ---------- functions ----------

function existscommand() {
    # コマンドの存在はこれで確認できる
    type -a $1 > /dev/null 2>&1
}

function anaconda_install() {
    SAVEBASEDIR=$1 # 保存先ディレクトリ, ホームディレクトリが楽
    INSTALLERURL=`python anaconda_installer.py --ostype $2`

    if ! existscommand wget; then
        # curlコマンドで取得したファイル名を指定したディレクトリに送る方法
        cd ${SAVEBASEDIR} && { curl -O ${INSTALLERURL} ; cd -; }
    else
        wget -P ${SAVEBASEDIR} ${INSTALLERURL}
    fi
}

# ---------- main programs -----------

if [ `uname -s` == 'Darwin' ]; then
    OS='MacOS'
else
    OSTYPENAME=`expr substr $(uname -s) 1 5`

    if [ ${OSTYPENAME} == 'MINGW' ]; then
        OS='Windows'
    elif [ ${OSTYPENAME} == 'Linux' ]; then
        OS='Linux'
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
            ${PROCESSFLAG}=1
            echo 'Start to install Anaconda3.'
        elif [ ${INSTALLFLAG} == 'no' ]; then
            ${PROCESSFLAG}=1
            echo 'Stop installing anaconda3, try again.'
        elif [ ${#INSTALLFLAG} -eq 0 ]; then
            ${PROCESSFLAG}=0
        else
            ${PROCESSFLAG}=0
        fi

        if [ ${PROCESSFLAG} -eq 1 ]; then
            break
        else
            continue
        fi
    done
else
    echo 'Anaconda existed, please check whether or not anaconda3.'
fi
