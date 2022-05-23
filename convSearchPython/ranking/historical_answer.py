"""Historical Answer Expansion (HAExp)
from Query and Answer Expansion from Conversation History(Jheng-Hong Yang et al)"""
import logging
from datetime import timedelta
from time import time

import pandas as pd
from torch.cuda import is_available as cuda_available

from convSearchPython.dataset import Conversations
from convSearchPython.pipelines import Reranker, StepType
from convSearchPython.ranking.bert.bert_pre_marco import DL4MBertClassifier
from convSearchPython.utils.data_utils import queries_conversation_splitter, limit_per_qid, replace_col_with_history, \
    sort_results_update_rank

logger = logging.getLogger(__name__)


def __sort_temp(t):
    return t[1]


def __generate_curr(group: pd.DataFrame):
    for _, df in group.iterrows():
        yield df['docno'], df['log_prob']


def __drop_dp(data: pd.DataFrame, *tpl):
    d = data[tpl]
    d.drop_duplicates(inplace=True)
    return d


def __haexp_rerank_conv(log_data: pd.DataFrame, _lambda: float,
                        k: int) -> pd.DataFrame:
    """
    Private function. Apply HAExp rerank to a single conversation

    Args:
        log_data: input df having an additional column 'log_prob' with the
        negative log-likelihood already calculated
        _lambda: decay factor
        k: cutoff

    Returns:
        re-ranked df
    """
    # ['qid', 'docid', 'docno', 'text', 'rank', 'score', 'query_0', 'query']
    cols = tuple(log_data.columns)
    cols_minus_rank = tuple(c for c in cols if c != 'rank')
    q_cols = tuple(c for c in cols if c.startswith('query'))
    unknown_cols = tuple(c for c in cols if c not in q_cols +
                         ('qid', 'docid', 'docno', 'text', 'rank', 'score', 'log_prob'))

    last = []
    temp = []
    parts = []
    for qid, group in log_data.groupby('qid'):
        temp.clear()
        curr = list(__generate_curr(group))
        curr_docno = set(pd.unique(group['docno']))
        temp.extend(curr)
        temp.extend((docno, log * _lambda) for docno, log in last if docno not in curr_docno)
        temp.sort(key=__sort_temp)
        last = curr

        _k = k if k > 0 else len(group)
        res = temp[:_k]

        col_dict = {c: [] for c in cols}
        col_dict['log_prob'] = []
        count = iter(range(1, len(res) + 1))
        ex_row = group.iloc[0]
        for docno, log in res:
            if docno in curr_docno:
                row = group[group['docno'] == docno].iloc[0]
                for c in cols_minus_rank:
                    col_dict[c].append(row[c])
                col_dict['rank'].append(next(count))
            else:
                row = log_data[log_data['docno'] == docno].iloc[0]
                col_dict['qid'].append(qid)
                col_dict['docid'].append(row['docid'])
                col_dict['docno'].append(docno)
                col_dict['text'].append(row['text'])
                col_dict['rank'].append(next(count))
                col_dict['score'].append(0)
                col_dict['log_prob'].append(log)
                for c in q_cols:
                    col_dict[c].append(ex_row[c])
                for c in unknown_cols:
                    col_dict[c].append(ex_row[c])

        g_res = pd.DataFrame(col_dict)
        # g_res.sort_values('rank', inplace=True)

        parts.append(g_res)

    result = pd.concat(parts)
    result.reset_index(inplace=True, drop=True)
    return result


def haexp_rerank(data: pd.DataFrame, conversations: Conversations, classifier: DL4MBertClassifier,
                 _lambda: float, k: int = None) -> pd.DataFrame:
    """
    Apply HAExp rerank to data

    Args:
        data: input data
        conversations: conversations structure
        _lambda: decay factor (should be >= 1)
        k: cutoff
        classifier: DL4MBertClassifier to use

    Returns:
        a new DataFrame with reranked results
    """
    if k is not None:
        data = limit_per_qid(data, k)
    else:
        data = data.copy()
        k = -1

    st = time()
    logger.info('Starting calculating log-likelihood with classifier %s', classifier.__class__)
    log_data = classifier.log_prob_df(data, inplace=True)
    logger.info('Log-likelihoods calculated in %s', timedelta(seconds=(time() - st)))

    parts = []
    for conv in queries_conversation_splitter(log_data, conversations):
        parts.append(__haexp_rerank_conv(conv, _lambda, k))
    return pd.concat(parts)


class HAEReranker(Reranker):
    """
    HAE reranker.

    Will use CUDA if available (highly recommended)
    """
    def __init__(self, conversations: Conversations, _lambda: float, k: int,
                 allow_cuda=True, **kwargs):
        """
        Args:
            conversations: conversations structure
            _lambda: decay factor (should be >= 1)
            k: cutoff
            allow_cuda: if cuda should be allowed
            **kwargs: extra unused arguments
        """
        super().__init__(**kwargs)
        self._conversations = conversations
        self._lambda = _lambda
        self._k = k

        self._cuda = cuda_available() and allow_cuda
        self._classifier = None

    @property
    def name(self) -> str:
        return f'HAE-l{self._lambda}-k{self._k}'

    @property
    def type(self) -> StepType:
        if self._cuda:
            return StepType.SEQUENTIAL
        else:
            return StepType.CONVERSATIONALLY_PARALLEL

    def cleanup(self):
        self._classifier = None

    def rerank(self, results: pd.DataFrame) -> pd.DataFrame:
        from convSearchPython.ranking.bert.bert_pre_marco import CpuDL4MBertClassifier, GpuDL4MBertClassifier
        if self._classifier is None:
            if self._cuda:
                self._classifier = GpuDL4MBertClassifier()
            else:
                self._classifier = CpuDL4MBertClassifier()
        results = haexp_rerank(results, self._conversations, self._classifier, self._lambda, self._k)

        # fix columns so it's consistent with pyterrier model
        replace_col_with_history('score', results['log_prob'], results, inplace=True)

        return sort_results_update_rank(results, self._conversations)


# if __name__ == '__main__':
#     _df = pd.read_pickle('test_results.pkl.gz')
#     # _df = _df[_df['qid'] == '31_1'][:100]
#     # _df = _df[_df['qid'] == '31_1']
#     _df = pd.concat([_df[_df['qid'] == '31_1'][:50], _df[_df['qid'] == '31_2'][:50]])
#     print(_df)
#     # DL4MBertClassifier().log_prob_df(_df, inplace=True)
#
#     _df2 = GpuDL4MBertClassifier().log_prob_df(_df, batch_size=5)
#     _rr = __haexp_rerank_conv(_df2, 2, -1)
#     print(_rr)
