mkdir bert_server && cd bert_server
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip chinese_L-12_H-768_A-12.zip 

bert-serving-start -model_dir chinese_L-12_H-768_A-12
 
