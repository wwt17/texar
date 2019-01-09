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

extractor_models: data2text/roto-convie_1 data2text/roto-convie_2 data2text/roto-convie_3 data2text/roto-blstmie_1 data2text/roto-blstmie_2 data2text/roto-blstmie_3

data2text/roto-convie_1 data2text/roto-convie_2 data2text/roto-convie_3 data2text/roto-blstmie_1 data2text/roto-blstmie_2 data2text/roto-blstmie_3:
	scp daoyuan:wwt/texar/examples/text_content_manipulation/data2text/$@ data2text/
