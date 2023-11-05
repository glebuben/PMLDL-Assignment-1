import torch
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator

nltk.download('punkt')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']


class CustomDataset(torch.utils.data.Dataset):
    """
    Class for custom dataset.

    It can be used with Custom models and with pretrained models from 'transformers' package
    """
    def __init__(self, dataframe: pd.DataFrame, vocab=None, tokenizer=None):
        """
        Creates Custom Dataset object
        :param dataframe: pandas dataframe with initial data
        :param vocab: vocabulary
        :param tokenizer: tokenizer for preprocessing
        """
        self.df = dataframe
        self._preprocess()
        self.tokenizer = tokenizer
        self.vocab = vocab or self._create_vocab()

    def _preprocess(self):
        """
        apply tokenization on the initial data
        """
        self.tokenized = self.df.copy()
        self.tokenized['reference'] = self.df['reference'].apply(self._preprocess_text)
        self.reference = self.tokenized['reference'].to_list()
        self.tokenized['translation'] = self.df['translation'].apply(self._preprocess_text)
        self.translation = self.tokenized['translation'].to_list()

    def _preprocess_text(self, text: str) -> list[str]:  # function for preprocessing dataframe
        """
        apply lowercase and tokenization for given text
        :param text: string to be modified
        :return: preprocessed text
        """
        text = text.lower()
        if self.tokenizer is not None:
            tokenized = self.tokenizer(text)['input_ids']
        else:
            tokenized = word_tokenize(text)
            tokenized.insert(0, '<bos>')
            tokenized.append('<eos>')
        return tokenized

    def _create_vocab(self):
        """
        creates vocabulary for given dataset
        :return: created vocabulary
        """
        # creates vocabulary that is used for encoding
        # the sequence of tokens (splitted sentence)
        vocab = build_vocab_from_iterator(self.reference + self.translation, min_freq=2, specials=special_symbols)
        vocab.set_default_index(UNK_IDX)
        return vocab

    def _get_reference(self, index: int) -> list:
        """
        getter for reference
        :param index: index
        :return: returns reference for given index
        """
        # retrieves sentence from dataset by index
        sent = self.reference[index]
        if self.tokenizer is None:
            return self.vocab(sent)
        else:
            return sent

    def _get_translation(self, index: int) -> list:
        """
        getter for translation
        :param index: index
        :return: returns translation for given index
        """
        # retrieves tags from dataset by index
        trans = self.translation[index]
        if self.tokenizer is None:
            return self.vocab(trans)
        else:
            return trans

    def __getitem__(self, index):
        """
        getter for items
        :param index: index
        :return: returns data in appropriate type according to tokenizer
        """
        if self.tokenizer is not None:
            return {"input_ids": self._get_reference(index), 'labels': self._get_translation(index)}
        else:
            return self._get_reference(index), self._get_translation(index)

    def __len__(self) -> int:
        """
        length of the dataset
        """
        return len(self.reference)
