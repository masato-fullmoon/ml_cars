#!/bin/bash

function existscommand() {
    type -a $1 > /dev/null 2>&1
}

# bashでの本実行スクリプトディレクトリパスの取得方法
BASEDIR=$(cd $(dirname $0); pwd)

if ! existscommand nvidia-smi; then
    echo 'nvidia-smi not existed...'
    exit 1;
else
    nvidia-smi -L > ${BASEDIR}/gpus.log
fi

exit 0;
