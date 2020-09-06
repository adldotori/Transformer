import os
import os.path as osp
import pandas as pd
import re
import pickle
import argparse

import torch
from soynlp.word import WordExtractor
from soynlp.tokenizer import LTokenizer
from torchtext.data import Field, Dataset, Example, Iterator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KorEngDataset():
    def __init__(self, args):
        self.args = args
        
        if not self._check_files():
            train_file1 = os.path.join(self.args.path, '1_구어체(1)_200226.xlsx')
            train_file2 = os.path.join(self.args.path, '1_구어체(2)_200226.xlsx')

            df1 = pd.read_excel(train_file1)
            df2 = pd.read_excel(train_file2)
            df = pd.concat([df1, df2])
            df = df[['원문', '번역문']]
        
        else:
            df = None
        train, valid, test = self._data_split(df)
        self.kor_tokenizer, self.eng_tokenizer = self._tokenize(df)
        self.kor_vocab, self.eng_vocab, self.kor_vocab_size, self.eng_vocab_size = self._build_vocab(df, self.kor_tokenizer, self.eng_tokenizer)

        if args.mode == 'train':
            if args.debug:
                self.train_data = self._generate_dataset(test, self.kor_vocab, self.eng_vocab)
                self.valid_data = self.train_data
                self.test_data = self.train_data
            else:
                self.train_data = self._generate_dataset(train, self.kor_vocab, self.eng_vocab)
                self.valid_data = self._generate_dataset(valid, self.kor_vocab, self.eng_vocab)
                self.test_data = self._generate_dataset(test, self.kor_vocab, self.eng_vocab)
            print('Finish generate dataset')

    def dataloader(self):
        train_iter, valid_iter, test_iter = Iterator.splits(
        (self.train_data, self.valid_data, self.test_data),
        sort_within_batch=True,
        sort_key=lambda x: len(x.kor),
        batch_size=self.args.batch_size,
        device=device)

        return train_iter, valid_iter, test_iter

    def get_vocab_size(self):
        return self.kor_vocab_size, self.eng_vocab_size

    def get_tokenizer(self):
        return self.kor_tokenizer

    def get_vocab(self):
        return self.kor_vocab, self.eng_vocab

    def _check_files(self):
        train_path = osp.join(self.args.path, 'train.csv')
        valid_path = osp.join(self.args.path, 'valid.csv')
        test_path = osp.join(self.args.path, 'test.csv')

        kor_token_path = osp.join(self.args.path, self.args.kor_token_path)
        eng_token_path = osp.join(self.args.path, self.args.eng_token_path)

        kor_vocab_path = osp.join(self.args.path, self.args.kor_vocab_path)
        eng_vocab_path = osp.join(self.args.path, self.args.eng_vocab_path)

        return osp.isfile(train_path) and osp.isfile(valid_path) and osp.isfile(test_path)\
           and osp.isfile(kor_token_path) and osp.isfile(eng_token_path)\
           and osp.isfile(kor_vocab_path) and osp.isfile(eng_vocab_path)

    def _get_tokenizer(self, df):
        """
        Generate a torkenizer by extracting words
        Args:
            dataframe: data corpus of one language
        Returns:
            tokenizer
        """
        word_extractor = WordExtractor()
        word_extractor.train(df)
        words = word_extractor.extract()
        print(f'length of words is {len(words)}')
        cohesion_scores = {word: score.cohesion_forward for word, score in words.items()}
        tokenizer = LTokenizer(scores=cohesion_scores)
        return tokenizer

    def _clean_text(self, text):
        """
        remove special characters from the input sentence to normalize it
        Args:
            text: (string) text string which may contain special character
        Returns:
            normalized sentence
        """
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`…》]', '', text)
        return text

    def _generate_dataset(self, df, kor, eng):
        # convert each row of DataFrame to torchtext 'Example' containing 'kor' and 'eng' Fields
        list_of_examples = [Example.fromlist(row.apply(lambda x: self._clean_text(x)).tolist(),
                                                fields=[('kor', kor), ('eng', eng)]) for _, row in df.iterrows()]

        # construct torchtext 'Dataset' using torchtext 'Example' list
        dataset = Dataset(examples=list_of_examples, fields=[('kor', kor), ('eng', eng)])
        return dataset

    def _data_split(self, df):

        train_path = osp.join(self.args.path, 'train.csv')
        valid_path = osp.join(self.args.path, 'valid.csv')
        test_path = osp.join(self.args.path, 'test.csv')

        if not (osp.isfile(train_path) and osp.isfile(valid_path) and osp.isfile(test_path)):
            print(f'length of dataset is {len(df)}')
            length = len(df)
            df = df.sample(frac=1).reset_index(drop=True)
            train = df[:length//10 * 8]
            valid = df[length//10 * 8:length//10 * 9]
            test = df[length//10 * 9:]

            train.to_csv(train_path, index=False)
            valid.to_csv(valid_path, index=False)
            test.to_csv(test_path, index=False)
        else:
            train = pd.read_csv(train_path)
            valid = pd.read_csv(valid_path)
            test = pd.read_csv(test_path)

        return train, valid, test

    def _tokenize(self, df):
        """
        Get Korean, English tokenizer
        Args:
            args: Argument
            df: Text Corpus
        Returns:
            Korean tokenizer, English tokenizer
        """
        kor_token_path = osp.join(self.args.path, self.args.kor_token_path)
        eng_token_path = osp.join(self.args.path, self.args.eng_token_path)

        if not (osp.isfile(kor_token_path) and osp.isfile(eng_token_path)):
            kor_tokenizer = self._get_tokenizer(df['원문'])
            eng_tokenizer = self._get_tokenizer(df['번역문'])

            with open(kor_token_path, 'wb') as f:
                pickle.dump(kor_tokenizer, f)
            with open(eng_token_path, 'wb') as f:
                pickle.dump(eng_tokenizer, f)

        else:
            with open(kor_token_path, 'rb') as f:
                kor_tokenizer = pickle.load(f)
            with open(eng_token_path, 'rb') as f:
                eng_tokenizer = pickle.load(f)

        print('Make kor, eng tokenizer')
        return kor_tokenizer, eng_tokenizer

    def _build_vocab(self, df, kor_tokenizer, eng_tokenizer):
        """
        Make Dataset and vocabulary dictionary
        Args:
            args: Argument
            df: Text Corpus
            kor_tokenizer: Korean tokenizer
            eng_tokenizer: English tokenizer
        Returns:
            dataset: paired dataset
            kor: Korean vocabulary dictionary
            eng: English vocabulary dictionary
        """
        kor_vocab_path = osp.join(self.args.path, self.args.kor_vocab_path)
        eng_vocab_path = osp.join(self.args.path, self.args.eng_vocab_path)

        if not (osp.isfile(kor_vocab_path) and osp.isfile(eng_vocab_path)):
            kor = Field(tokenize = kor_tokenizer.tokenize,
                        lower = True)
            eng = Field(tokenize = eng_tokenizer.tokenize,
                        init_token = '<sos>',
                        eos_token = '<eos>',
                        lower = True)

            dataset = self._generate_dataset(df, kor, eng)

            kor.build_vocab(dataset, min_freq = 2)
            eng.build_vocab(dataset, min_freq = 2)

            with open(kor_vocab_path, 'wb') as f:
                pickle.dump(kor, f)

            with open(eng_vocab_path, 'wb') as f:
                pickle.dump(eng, f)
        else:
            with open(kor_vocab_path, 'rb') as f:
                kor = pickle.load(f)

            with open(eng_vocab_path, 'rb') as f:
                eng = pickle.load(f) 

        print('-'*90)
        print('Korean Vocabulary Dictioinary : ', kor.vocab.itos[:20])
        print('Commonly used words : ', kor.vocab.freqs.most_common(5))
        print('Length : ', len(kor.vocab))
        print('-'*90)
        print('English Vocabulary Dictioinary : ', eng.vocab.itos[:20])
        print('Commonly used words : ', eng.vocab.freqs.most_common(5))
        print('Length : ', len(eng.vocab))
        print('-'*90)

        return kor, eng, len(kor.vocab), len(eng.vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Builder')
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--kor_token_path', type=str, default='kor_token.pkl')
    parser.add_argument('--eng_token_path', type=str, default='eng_token.pkl')
    parser.add_argument('--kor_vocab_path', type=str, default='kor.pkl')
    parser.add_argument('--eng_vocab_path', type=str, default='eng.pkl')
    parser.add_argument('-b', '--batch_size', type=int, default=30)
    args = parser.parse_args()

    dataset = KorEngDataset(args)
    print(dataset.dataloader())