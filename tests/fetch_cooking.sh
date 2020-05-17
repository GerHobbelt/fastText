#!/usr/bin/env bash
#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

DATADIR=${DATADIR:-data}

report_error() {
   echo "Error on line $1 of $0"
}

myshuf() {
  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
}

normalize_text() {
  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
    sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
        -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
}

set -e
trap 'report_error $LINENO' ERR

mkdir -p "${DATADIR}"


echo "Downloading cooking dataset"

data_result="${DATADIR}"/cooking/cooking.stackexchange.txt
if [ ! -f "$data_result" ]
then
  mkdir -p "${DATADIR}"/cooking/
  wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz -O "${DATADIR}"/cooking/cooking.stackexchange.tar.gz
  tar xvzf "${DATADIR}"/cooking/cooking.stackexchange.tar.gz --directory "${DATADIR}"/cooking || exit 1
  cat "${DATADIR}"/cooking/cooking.stackexchange.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > "${DATADIR}"/cooking/cooking.preprocessed.txt
fi

data_result="${DATADIR}"/cooking.train
if [ ! -f "$data_result" ]
then
  head -n 12404 "${DATADIR}"/cooking/cooking.preprocessed.txt > "${DATADIR}"/cooking.train.txt
fi

data_result="${DATADIR}"/cooking.valid
if [ ! -f "$data_result" ]
then
  tail -n 3000 "${DATADIR}"/cooking/cooking.preprocessed.txt > "${DATADIR}"/cooking.valid.txt
fi