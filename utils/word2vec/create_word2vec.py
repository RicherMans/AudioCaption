# -*- coding: utf-8 -*-

from gensim.models import word2vec, doc2vec
import pandas as pd
import fire


def prepare_sentences(caption_json_path: str):
    df = pd.read_json(caption_json_path)
    tokens = df.tokens
    sentences = []
    for token in tokens:
        sentences.append(token)
    return sentences


def prepare_word2vec(sentences, embedding_size:int, model_path:str):
    model = word2vec.Word2Vec(sentences, min_count=1, size=embedding_size)
    model.save(model_path)


def prepare_doc2vec(sentences, embedding_size:int, model_path:str):
    documents = [doc2vec.TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    model = doc2vec.Doc2Vec(documents, vector_size=embedding_size, min_count=1, workers=4)
    model.save(model_path)


def main(caption_json_path, word_embedding_size, word2vec_path, sent_embedding_size, doc2vec_path):
    sentences = prepare_sentences(caption_json_path)
    prepare_word2vec(sentences, word_embedding_size, word2vec_path)
    prepare_doc2vec(sentences, sent_embedding_size, doc2vec_path)


if __name__ == '__main__':
    fire.Fire(main)
