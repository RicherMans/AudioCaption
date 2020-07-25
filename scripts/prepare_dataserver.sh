#!/bin/bash

if [ $# -lt 1 ]; then
  cat >&2 <<EOF 
Usage: $0 <dataserver-dir> 
e.g.: $0 utils/dataserver
EOF
  exit 1;
fi

dataserver=$1

[ ! -d $dataserver ] && mkdir $dataserver

cd $dataserver

[ ! -d stanford-corenlp-full-2018-10-05 ] && wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip && unzip stanford-corenlp-full-2018-10-05.zip

cd stanford-corenlp-full-2018-10-05

[ ! -f stanford-chinese-corenlp-2018-10-05-models.jar ] && wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-10-05-models.jar

[ ! -f StanfordCoreNLP-chinese.properties ] && wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties


java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-serverProperties StanfordCoreNLP-chinese.properties \
-preload tokenize,ssplit,pos,lemma,ner,parse \
-status_port 9000 -port 9000 -timeout 15000
