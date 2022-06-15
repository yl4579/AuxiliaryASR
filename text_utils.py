#coding:utf-8
import os
import os.path as osp
import pandas as pd

DEFAULT_DICT_PATH = osp.join('word_index_dict.txt')
class TextCleaner:
    def __init__(self, word_index_dict_path=DEFAULT_DICT_PATH):
        self.word_index_dictionary = self.load_dictionary(word_index_dict_path)

    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(char)
        return indexes

    def load_dictionary(self, path):
        csv = pd.read_csv(path, header=None).values
        word_index_dict = {word: index for word, index in csv}
        return word_index_dict
