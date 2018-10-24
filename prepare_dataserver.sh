mkdir dataserver
cd dataserver
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip
unzip stanford-corenlp-full-2018-02-27.zip && cd stanford-corenlp-full-2018-02-27
wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-02-27-models.jar
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties 

java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-serverProperties StanfordCoreNLP-chinese.properties \
-preload tokenize,ssplit,pos,lemma,ner,parse \
-status_port 9000 -port 9000 -timeout 15000
