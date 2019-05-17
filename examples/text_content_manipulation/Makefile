all: nba_data data2text

nba_data:
	git clone https://github.com/ZhitingHu/text_content_manipulation.git && mv text_content_manipulation nba_data

data2text: rotowire roto-ie res extractor_models

rotowire: data2text/rotowire

data2text/rotowire:
	cd data2text && wget https://raw.githubusercontent.com/harvardnlp/boxscore-data/master/rotowire.tar.bz2; tar xjvf rotowire.tar.bz2

roto-ie: data2text/roto-ie.h5 data2text/roto-ie.dict data2text/roto-ie.labels

data2text/roto-ie.h5 data2text/roto-ie.dict data2text/roto-ie.labels:
	cd data2text && python3 data_utils.py make_ie_data --input_path rotowire --output roto-ie.h5

res: data2text/res

data2text/res:
	cd data2text && mkdir res && python3 data_utils.py extract_sent --output res/

extractor_models: data2text/roto-convie_1 data2text/roto-convie_2 data2text/roto-convie_3 data2text/roto-blstmie_1 data2text/roto-blstmie_2 data2text/roto-blstmie_3

data2text/roto-convie_1 data2text/roto-convie_2 data2text/roto-convie_3 data2text/roto-blstmie_1 data2text/roto-blstmie_2 data2text/roto-blstmie_3:
