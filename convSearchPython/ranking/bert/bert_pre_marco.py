"""Load pre-trained BERT (dl4marco-bert)"""
import gc
import logging
from abc import abstractmethod
from os import environ

import numpy as np
import pandas as pd
import torch
import torch.cuda as cuda
from torch.nn import DataParallel
from transformers import BertForSequenceClassification, BertTokenizerFast

CACHE = {'batch_size': 80}


# torch.use_deterministic_algorithms(True)

if 'NO_PRELOAD' not in environ:
    # make sure required data is downloaded when this module is loaded
    BertForSequenceClassification.from_pretrained('castorini/monobert-large-msmarco-finetune-only')


class DL4MBertClassifier:
    @abstractmethod
    def log_prob_df(self, *args, **kwargs) -> pd.DataFrame:
        """Compute the log-likelihood between queries and doc in a dataset"""
        raise NotImplementedError()


class CpuDL4MBertClassifier(DL4MBertClassifier):
    """Classifier for the dl4marco-bert model pre-trained on MSMARCO"""

    def __init__(self):
        self.model = BertForSequenceClassification.from_pretrained('castorini/monobert-large-msmarco-finetune-only') \
            .eval()
        self.tokenizer = BertTokenizerFast.from_pretrained('castorini/monobert-large-msmarco-finetune-only')

    def log_prob(self, query: str, doc: str) -> float:
        """Compute the log-likelihood probability between a query and a document

        :param query: the query text
        :param doc: the document text
        :return: a float in [0,1] representing the log-likelihood probability"""
        encoding = self.tokenizer(query, doc, return_tensors="pt", truncation='only_second')
        logits = self.model(**encoding).logits
        sft = torch.softmax(logits, dim=1)
        return (- torch.log(sft[0][1])).item()

    def log_prob_df(self, data: pd.DataFrame, inplace=False) -> pd.DataFrame:
        """Compute the log-likelihood between queries and doc in a dataset.

        :param data: input df
        :param inplace: if modify data or create a copy
        :return: df with 'log_prob' column"""
        if not inplace:
            data = data.copy()
        data['log_prob'] = np.vectorize(self.log_prob)(data['query'], data['text'])
        return data


class GpuDL4MBertClassifier(DL4MBertClassifier):
    """Classifier for the dl4marco-bert model pre-trained on MSMARCO"""

    def __init__(self):
        model = BertForSequenceClassification.from_pretrained('castorini/monobert-large-msmarco-finetune-only')
        model = model.eval()
        model = DataParallel(model.to('cuda:0'))
        self.model = model
        self.tokenizer = BertTokenizerFast.from_pretrained('castorini/monobert-large-msmarco-finetune-only')
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info('Using cuda: %s', cuda.get_device_name(cuda.current_device()))

    def log_prob(self, query: str, doc: str) -> float:
        """Compute the log-likelihood probability between a query and a document

        :param query: the query text
        :param doc: the document text
        :return: a float in [0,1] representing the log-likelihood probability"""
        encoding = self.tokenizer(query, doc, return_tensors="pt", truncation='only_second')
        logits = self.model(**encoding).logits
        sft = torch.softmax(logits, dim=1)
        lg = - torch.log(sft[0][1])
        return lg.item()

    def _log_prob_batch(self, batch_a, batch_b):
        try:
            # return True, np.fromiter(
            #     ((-torch.log(s[1])).item() for s in torch.softmax(
            #         self.model(**(
            #             self.tokenizer(batch_a, batch_b, return_tensors="pt", truncation='only_second', padding=True)
            #         )).logits, dim=1
            #     )), float, len(batch_a)
            # )
            # return True, np.fromiter(
            #     (ls[1].item() for ls in -torch.log_softmax(
            #         self.model(**(
            #             self.tokenizer(batch_a, batch_b, return_tensors="pt", truncation='only_second', padding=True)
            #         )).logits, dim=1
            #     )), float, len(batch_a)
            # )
            return True, np.fromiter(
                (item[1] for item in (-torch.log_softmax(
                    self.model(**(
                        self.tokenizer(batch_a, batch_b, return_tensors="pt", truncation='only_second', padding=True)
                    )).logits, dim=1)
                ).detach().cpu().numpy()), float, len(batch_a)
            )
        except RuntimeError as ex:  # Cuda Out Of Memory
            self.__logger.warning(ex)
            return False, str(ex)

    def _log_prob_loop(self, batch_size: int, queries: list, size_reduce: int, texts: list):
        size = len(queries)
        count = 0
        results = []
        while count < size:
            ok, res = self._log_prob_batch(
                queries[count:count + batch_size],
                texts[count:count + batch_size]
            )
            if not ok:
                if batch_size == 1:
                    self.__logger.error('Runtime error during computation: %s', res)
                    raise Exception()
                new_size = batch_size - size_reduce
                new_size = new_size if new_size >= 1 else 1
                self.__logger.warning('Trying to reduce batch size from %d to %d', batch_size, new_size)
                batch_size = new_size
                gc.collect()  # force garbage collection
                continue
            results.append(res)
            count = count + batch_size
        CACHE['batch_size'] = batch_size
        return np.concatenate(results, dtype=float)

    def log_prob_df(self, data: pd.DataFrame, inplace=False, batch_size: int = None, size_reduce=5) -> pd.DataFrame:
        """Compute the log-likelihood between queries and doc in a dataset.

        :param data: input df
        :param inplace: if modify data or create a copy
        :param batch_size: number of rows to compute at time
        :param size_reduce: how much reduce batch_size if computation fail for OOM
        :return: df with 'log_prob' column"""
        if not inplace:
            data = data.copy()

        if batch_size is None:
            batch_size = CACHE['batch_size']

        queries = list(data['query'])
        texts = list(data['text'])
        log_col = self._log_prob_loop(batch_size, queries, size_reduce, texts)

        data['log_prob'] = log_col
        return data


if __name__ == '__main__':
    _classifier = GpuDL4MBertClassifier()
    _query = "Good Morning at everyone!"
    _docs = ['Good day', 'Bad week', 'Everyone listen!', 'A good morning for everyone.', _query]
    _docs.append(' '.join('HI' for _ in range(513)))

    print(f'Likelihood with query "{_query}"')
    _df = pd.DataFrame()
    _df['text'] = _docs
    _df['query'] = [_query for _ in range(len(_docs))]
    _df2 = _classifier.log_prob_df(_df, batch_size=5)
    print(_df2)
    print(_df2.iloc[0]['log_prob'])
    print(type(_df2.iloc[0]['log_prob']))
    print(type(_df2.iloc[0]['query']))

    _df3 = CpuDL4MBertClassifier().log_prob_df(_df)
    print(_df3)

    print('END')
