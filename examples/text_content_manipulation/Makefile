all: nba_data data2text

nba_data: nba_data/gold.valid.txt nba_data/gold.test.txt

nba_data/gold.valid.txt nba_data/gold.test.txt:
	cd nba_data; python join_data.py

data2text: rotowire roto-ie extractor_models

rotowire: data2text/rotowire

data2text/rotowire:
	cd data2text; wget https://raw.githubusercontent.com/harvardnlp/boxscore-data/master/rotowire.tar.bz2; tar xjvf rotowire.tar.bz2

roto-ie: data2text/roto-ie.h5 data2text/roto-ie.dict data2text/roto-ie.labels

data2text/roto-ie.h5 data2text/roto-ie.dict data2text/roto-ie.labels:
	cd data2text; python data_utils.py -mode make_ie_data -input_path rotowire -output_fi roto-ie.h5

extractor_models: data2text/roto-convie-ep3-96-49 data2text/roto-convie-ep5-95-64 data2text/roto-convie-ep9-95-69 data2text/roto-blstmie-ep2-95-70 data2text/roto-blstmie-ep9-91-79 data2text/roto-blstmie-ep3-94-72

data2text/roto-convie-ep3-96-49 data2text/roto-convie-ep5-95-64 data2text/roto-convie-ep9-95-69 data2text/roto-blstmie-ep2-95-70 data2text/roto-blstmie-ep9-91-79 data2text/roto-blstmie-ep3-94-72:
	scp tir:zichaoy/lithium_project/zichaoy/$@ data2text/
