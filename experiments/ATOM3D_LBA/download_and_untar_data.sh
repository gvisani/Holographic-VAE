
datadir=./data/

mkdir -p $datadir

echo downloading data ...

curl https://zenodo.org/record/4914718/files/LBA-raw.tar.gz?download=1 > $datadir"LBA-raw.tar.gz"

curl https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30-indices.tar.gz?download=1 > $datadir"LBA-split-by-sequence-identity-30-indices.tar.gz"

curl https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-60-indices.tar.gz?download=1 > $datadir"LBA-split-by-sequence-identity-60-indices.tar.gz"

echo done downloading, now untarring ...

tar -xvf $datadir"LBA-raw.tar.gz" -C $datadir
mv $datadir"raw" $datadir"LBA-raw"

tar -xvf $datadir"LBA-split-by-sequence-identity-30-indices.tar.gz" -C $datadir
mv $datadir"indices" $datadir"LBA-split-by-sequence-identity-30-indices"

tar -xvf $datadir"LBA-split-by-sequence-identity-60-indices.tar.gz" -C $datadir
mv $datadir"indices" $datadir"LBA-split-by-sequence-identity-60-indices"

echo deleting tar files ...

rm $datadir"LBA-raw.tar.gz"
rm $datadir"LBA-split-by-sequence-identity-30-indices.tar.gz"
rm $datadir"LBA-split-by-sequence-identity-60-indices.tar.gz"

