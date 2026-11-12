#!/bin/sh
# Download MovieLens 25M (about 250 MB) from GroupLens into realworldexample/data/ml-25m and check its MD5.
# The data are not in this repository; see https://grouplens.org/datasets/movielens/25m/ for the license.
set -e
cd "$(dirname "$0")"
mkdir -p data && cd data
curl -fL -o ml-25m.zip https://files.grouplens.org/datasets/movielens/ml-25m.zip
curl -fL -o ml-25m.zip.md5 https://files.grouplens.org/datasets/movielens/ml-25m.zip.md5
expected=$(cut -d' ' -f1 ml-25m.zip.md5)
actual=$( (md5sum ml-25m.zip 2>/dev/null || md5 -r ml-25m.zip) | cut -d' ' -f1)
[ "$expected" = "$actual" ] || { echo "MD5 mismatch: expected $expected, got $actual"; exit 1; }
unzip -oq ml-25m.zip ml-25m/ratings.csv ml-25m/movies.csv ml-25m/README.txt
echo "MovieLens 25M is in $(pwd)/ml-25m (MD5 $actual)"
