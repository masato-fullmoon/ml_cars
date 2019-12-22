#!/bin/bash

# ---------- functions ----------

function CommandExistance() {
    # コマンドが使えるのかのチェック
    type -a $1 > /dev/null 2>&1
}

function ErrorMassage() {
    # エラーメッセージを赤文字で標準出力, ステータスに1を返す
    echo -e "\033[0;31m$1\033[0;39m"
    return 1
}

function SuccessMassage() {
    # 成功メッセージをシアンで出力, ステータスに0を返す
    echo -e "\033[0;36m$1\033[0;39m"
    return 0
}

function NormalsMassage() {
    # 通常メッセージを緑字で出力
    echo -e "\033[0;32m$1\033[0;39m"
}

function WarniningMassage() {
    # 警告メッセージを黄色字で出力
    echo -e "\033[0;33m$1\033[0;39m"
}

function Anaconda3Install() {
    # -------------
    # argv1: osの種類
    # argv2: インストーラーファイルへのパス
    # -------------

    if [ $1 == 'Linux' ]; then
        bash $2
    else
        ErrorMassage "Sorry, Linux only. Test now."
        exit
    fi
}

# ---------- global vars ----------

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
        exit
    fi
fi

ANACONDASAVEDIR=~
ANACONDADLURL='https://www.anaconda.com/distribution/'

URLHREFS=`curl "${ANACONDADLURL}" | grep 'Anaconda3' | grep 'href'`

# ---------- main programs ----------

# 空配列の宣言
declare -a urlarray=()

for href in ${URLHREFS}
do
    if [ `echo "${href}" | grep 'href' | grep "${OS}"` ]; then
        # ダブルクォーテーションの場合で囲まれた部分を抽出
        url=`echo "${href}" | sed 's/^.*"\(.*\)".*$/\1/'`
        # シングルクォーテーションの場合で囲まれた部分を抽出
        #url=`echo "${href}" | sed "s/^.*"\(.*\)".*$/\1/"`
        urlarray=( "${urlarray[@]}" "${url}" )
    fi
done

# 配列のインデックスを参照しつつ展開する配列のfor文
declare -a idxarray=()

for i in "${!urlarray[@]}"
do
    WarniningMassage "Index: ${i} => [${urlarray[${i}]}]"
    idxarray=( "${idxarray[@]}" "${i}" )
done

# bashで算術計算するときは以下のようにするといい
lastidx=`echo "$((${#urlarray[@]}-1))"`

if [ ${lastidx} -eq -1 ]; then
    lastidx=0
fi

while :
do
    read -p "Select integer index of DL URLs [from 0 to ${lastidx}] >>> " urltype

    if [ -z "${urltype}" ]; then
        continue
    else
        if [ ! `echo ${idxarray[@]} | grep ${urltype}` ]; then
            continue
        else
            installerurl="${urlarray[${urltype}]}"
            SuccessMassage "********** Start to install [`basename ${installerurl}`] **********"

            cd ${ANACONDASAVEDIR}

            if ! CommandExistance wget; then
                NormalsMassage '---------- Use command: [curl] ----------'
                curl -O ${installerurl}
            else
                NormalsMassage '---------- Use command: [wget] ----------'
                wget ${installerurl}
            fi

            if [ $? -eq 0 ]; then
                SuccessMassage '----------- Anaconda3 Installer Acomplished. ----------'
                installer=${ANACONDASAVEDIR}/`basename "${installerurl}"`

                break
            else
                ErrorMassage '----------- Anaconda3 Installer Failed. Try again. ----------'
                exit
            fi
        fi
    fi
done

while :
do
    read -p "Start to install Anaconda3 [yes/no] >>> " startflag

    if [ -z ${startflag} ]; then
        continue
    else
        if [ ${startflag} == 'yes' ]; then
            SuccessMassage "Activate Installation [${installer}]"
            Anaconda3Install ${OS} ${installer}

            WarniningMassage "Start to remove installer file [${installer}]"
            rm ${installer}

            if [ $? -eq 0 ]; then
                SuccessMassage "Anaconda3 Installation Acomplished."
                break
            else
                ErrorMassage "Anaconda3 Installation Failed. Try again."
                exit
            fi
        elif [ ${startflag} == 'no' ]; then
            ErrorMassage 'Stop to install Anaconda3. Try again.'
            exit
        else
            continue
        fi
    fi
done
