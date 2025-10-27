#!/bin/bash

mkdir -p graphs
cd graphs

wget -O datasets.zip "https://zenodo.org/records/16733396/files/archive.zip?download=1"
unzip datasets.zip -d .

cd ..


echo "Done"

