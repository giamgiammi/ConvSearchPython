"""Load CAsT dataset"""
from enum import Enum
from functools import partial
from typing import Tuple, Dict, List, Union

import pandas as pd
from pyterrier.datasets import DATASET_MAP, RemoteDataset
from pyterrier.io import SUPPORTED_TOPICS_FORMATS

from convSearchPython.basics import pyterrier as pt
from convSearchPython.dataset import Conversations, QueryMap


def _read_topics_json(filename, tag='raw_utterance', tokenise=True):
    from jnius import autoclass
    import json
    tokeniser = autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

    data = json.load(open(filename))

    topics = []
    for turn in data:
        turn_id = str(turn['number'])
        for utt in turn['turn']:
            utt_id = str(utt['number'])
            utt_text = utt[tag]
            if tokenise:
                utt_text = " ".join(tokeniser.getTokens(utt_text))
            #            topics.append((turn_id + '_' + utt_id, utt_text, turn_id, utt_id))
            topics.append((turn_id + '_' + utt_id, utt_text))
    #    return pd.DataFrame(topics, columns=["qid", "query", "tid", "uid"])
    return pd.DataFrame(topics, columns=["qid", "query"])


def _load():
    TREC_CAST = {
        "topics": {
            "original-2019": ("evaluation_topics_v1.0.json",
                              "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json",
                              "json_raw"),
            "resolved-2019": ("evaluation_topics_annotated_resolved_v1.0.tsv",
                              "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv",
                              "singleline"),

            "original-2020": ("2020_manual_evaluation_topics_v1.0.json",
                              "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
                              "json_raw"),
            "resolved-2020": ("2020_manual_evaluation_topics_v1.0.json",
                              "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json",
                              "json_manual"),

        },
        "qrels": {
            "2019": ("cast_eval_topics_2019.qrels", ""),
            "2020": ("cast_eval_topics_2020.qrels", ""),
        },
        "info_url": "https://github.com/daltonj/treccastweb",
    }
    DATASET_MAP['cast'] = RemoteDataset("CAST", TREC_CAST)
    SUPPORTED_TOPICS_FORMATS['json_raw'] = partial(_read_topics_json, tag='raw_utterance')
    SUPPORTED_TOPICS_FORMATS['json_manual'] = partial(_read_topics_json, tag='manual_rewritten_utterance')


_load()


class CastDataset(Enum):
    """
    Represent a loadable CaST Dataset
    """
    cast2019 = '2019'
    cast2020 = '2020'

    def get_dataset(self):
        """
        Get the relative dataset
        Returns:
            A tuple of (queries, qrels, conversations, query_map).

            Refer to cast_dataset function for more information.
        """
        return cast_dataset(self)


def cast_dataset(dataset: Union[CastDataset, str, int]) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                 Conversations, QueryMap]:
    """
    Return cast 2019/2020 datasets
    Args:
        dataset: the dataset to load. Can be a CastDataset enum, or a year

    Returns:
        A tuple composed of (queries, qrels, conversations, query_map).
         queries: the dataset as DataFrame
         qrels: the qrels as DataFrame
         conversations: a map of conversation conv_id -> list of qid
         query_map: a map of queries qid -> (conv_id, index)
    """
    if isinstance(dataset, CastDataset):
        year = dataset.value
    else:
        year = str(dataset)

    queries = pt.get_dataset("cast").get_topics(f'original-{year}')
    qrels = pt.get_dataset("cast").get_qrels(year)

    conversations = {}
    query_map = {}
    for _, row in queries.iterrows():
        qid = row['qid']
        conv_id = qid[0:qid.index('_')]
        q_list = conversations.get(conv_id)
        if q_list is None:
            q_list = []
            conversations[conv_id] = q_list
        q_list.append(qid)
        index = len(q_list) - 1
        query_map[qid] = (conv_id, index)
    return queries, qrels, conversations, query_map
