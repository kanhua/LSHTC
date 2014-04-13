mkdir source\ data
mkdir submission
mkdir meta\ data

wget https://www.dropbox.com/s/7lxk53jzlasu6d0/test-remapped.csv
wget https://www.dropbox.com/s/rne4vv8qhc12ozs/test-remapped-min.csv
wget https://www.dropbox.com/s/v87d2wkd1lhu9ny/train-sklearn.csv
wget https://www.dropbox.com/s/4zt9an06dyxs8xk/test-sklearn.csv
wget https://www.dropbox.com/s/belgvetaefezgo6/train-sk-min.csv
wget https://www.dropbox.com/s/2glzuiapn43qnz4/test-sk-min.csv

mv test-remapped.csv ./source\ data/
mv test-remapped-min.csv ./source\ data/
mv train-sklearn.csv ./source\ data/
mv test-sklearn.csv ./source\ data/
mv train-sk-min.csv ./source\ data/
mv test-sk-min.csv ./source\ data/