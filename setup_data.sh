cd data/dvd

tar -xvzf dialogs.tar.gz
cat monet_feats_part* > monet_feats.tar.gz
tar -xvzf monet_feats.tar.gz

rm dialogs.tar.gz
rm monet_feats.tar.gz
rm monet_feats_part00.tar.gz
rm monet_feats_part01.tar.gz

cd ../simmc
tar -xvzf dialogs.tar.gz
rm dialogs.tar.gz

cd ../..
