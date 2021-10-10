#!/usr/bin/env bash
# Download Pascal VOC 2007 + 2012 to data/voc.
set -euo pipefail

DEST="${1:-data/voc}"
mkdir -p "$DEST"
cd "$DEST"

if [ ! -d VOCdevkit/VOC2007 ]; then
    echo "fetching VOC2007..."
    curl -L -O http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    curl -L -O http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    tar xf VOCtrainval_06-Nov-2007.tar
    tar xf VOCtest_06-Nov-2007.tar
    rm -f VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar
fi

if [ ! -d VOCdevkit/VOC2012 ]; then
    echo "fetching VOC2012..."
    curl -L -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar xf VOCtrainval_11-May-2012.tar
    rm -f VOCtrainval_11-May-2012.tar
fi

echo "done. data is in $(pwd)"
