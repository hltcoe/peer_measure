from typing import Dict, List, Set, Tuple
import ir_measures
from ir_measures import providers, measures, Metric
from ir_measures.providers.base import Any, NOT_PROVIDED

import numpy as np
import scipy.stats as sts

def kruskal(*samples):
    # np.nan are unretrieved docs that do not have a rank, should tack at the end of each sample
    # with an average rank

    if np.isnan(np.concatenate(samples)).all():
        return np.nan, 1.0
    
    max_rank = np.nanmax(sum(samples, [])) if np.isnan(np.concatenate(samples)).all() is None else sum(map(len, samples))
    n_unretrieved = list(map(np.sum, map(np.isnan, map(np.asarray, samples))))
    unretrieved_rank = (2*max_rank+sum(n_unretrieved)+1)/2

    samples = [
        [ r for r in org if not np.isnan(r) ] + [unretrieved_rank]*addition
        for org, addition in zip(samples, n_unretrieved)
    ]
        
    # remove empty group
    samples = list(map(np.asarray, [ s for s in samples if len(s) > 0 ]))

    num_groups = len(samples)
    if num_groups < 2:
        # define as 1.0 because only ranking docs from that lang is the right thing to do
        return (np.nan, 1.0)
        
    n = np.asarray(list(map(len, samples)))
    alldata = np.concatenate(samples)
    totaln = n.sum(dtype=float)
    
    r_bar = alldata.mean()
    group_r_bar = np.asarray( list(map(np.sum, samples)) ) / n
    ssg = np.square(group_r_bar - r_bar) @ n
    ssa = np.square(alldata - r_bar).sum()

    h = (totaln - 1) * ssg / ssa
    df = num_groups - 1

    return h, sts.distributions.chi2.sf(h, df)


class _PEER(measures.Measure):
    """
    The Probability of documents in different languages (or groups) with Equal Expected Rank in the top k results.

<cite>
@inproceedings{10.1145/3626772.3657943,
    author={Eugene Yang and Thomas Jänich and James Mayfield and Dawn Lawrie},
    title={Language Fairness in Multilingual Information Retrieval},
    year={2023},
    url={https://doi.org/10.1145/3626772.3657943},
    doi={10.1145/3626772.3657943},
    booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    series={SIGIR '24}
}
</cite>
    """
    __name__ = 'PEER'
    NAME = __name__
    PRETTY_NAME = 'PEER at k'
    SHORT_DESC = 'The Probability of documents in different languages (or groups) with Equal Expected Rank in the top k results.'
    SUPPORTED_PARAMS = {
        'cutoff': measures.ParamInfo(dtype=int, required=True, desc='ranking cutoff threshold'),
        'weights': measures.ParamInfo(dtype=dict, required=True, desc='custom weight mapping (int-to-float)'),
        'lang_mapping': measures.ParamInfo(dtype=dict, required=True, desc='doc id to language mapping (str-to-str)')
    }

    def validate_params(self):
        super().validate_params() # self.validated has already turned True, do more assertion
        assert np.isclose(sum(self.params['weights'].values()), 1.0), "Weights should sum up to 1.0"

    def __repr__(self):
        result = self.__name__
        params = ','.join(
            f'{k}={self._param_repr(v)}'
            for k, v in self.params.items() 
            if k != self.AT_PARAM and v != self.SUPPORTED_PARAMS[k].default and k != 'lang_mapping'
        )
        if params:
            result = f'{result}({params})'
        if self.AT_PARAM in self.params:
            result = f'{result}@{self.params[self.AT_PARAM]}'
        return result


PEER = _PEER()
measures.register(PEER)


class PEERProvider(providers.Provider):
    """
    python implementation of PEER
    """
    NAME = 'peer_measure'
    SUPPORTED_MEASURES = [
        PEER(cutoff=Any(), lang_mapping=Any(), weights=Any())
    ]

    def _evaluator(self, measures, qrels):
        meta: Dict[int, List[Tuple]] = {}
        all_mappings: Dict[int, Dict[str, str]] = {}
        for measure in ir_measures.util.flatten_measures(measures):
            if measure.NAME == 'PEER':
                measure
                if id(measure['lang_mapping']) not in meta:
                    meta[id(measure['lang_mapping'])] = []
                    all_mappings[id(measure['lang_mapping'])] = measure['lang_mapping']
                meta[id(measure['lang_mapping'])].append((measure['cutoff'], measure))
            else:
                raise ValueError(f'unsupported measure {measure}')
        qrels = ir_measures.util.QrelsConverter(qrels).as_dict_of_dict()
        return PEEREvaluator(measures, qrels, meta, all_mappings)


class PEEREvaluator(providers.Evaluator):
    def __init__(self, measures, qrels, meta, all_mappings):
        super().__init__(measures, set(qrels.keys()))
        self.qrels: Dict[str, Dict[str, int]] = qrels
        self.meta: Dict[int, List[Tuple]] = meta
        self.all_mappings: Dict[int, Tuple[Set[str], Dict[str, str]]] = {
            mid: [set(mapping.values()), mapping]
            for mid, mapping in all_mappings.items()
        }

    def assign_rels_langs_ranks(
            self, 
            mapping_id: int, cutoff: int, 
            qrel: Dict[str, int], run: List[Tuple[str, float]]
        ):# -> dict[str, list] | None:
        ret = { 
            rel: { lang: [] for lang in self.all_mappings[mapping_id][0] } 
            for rel in qrel.values()
        }
        mapping = self.all_mappings[mapping_id][1]
        run = run[:cutoff]
        for i, (doc_id, _) in enumerate(run):
            ret[ qrel.get(doc_id, 0) ][ mapping[doc_id] ].append( i+1 )
        
        # add unretrieve rel docs
        retrieved_docs = set(doc_id for doc_id, _ in run)
        for doc_id, rel in qrel.items():
            if rel > 0 and doc_id not in retrieved_docs:
                ret[rel][mapping[doc_id]].append(np.nan)
                
        return ret

    def _iter_calc(self, run):
        run = ir_measures.util.RunConverter(run).as_dict_of_dict()
        sorted_run = {q: list(sorted(run[q].items(), key=lambda x: (-x[1], x[0]))) for q in run}
        for qid in run:
            qid_qrels = self.qrels.get(qid)
            if not(qid_qrels):
                continue

            current_run = sorted_run.get(qid, [])
            for mapping_id, cutoff_measures in self.meta.items():
                for cutoff, measure in cutoff_measures:
                    rels_langs_ranks = self.assign_rels_langs_ranks(mapping_id, cutoff, qid_qrels, current_run)
                    cutoff = cutoff if cutoff != NOT_PROVIDED else len(current_run) 
                    weights: Dict[int, float] = measure['weights']
                    
                    value = sum([
                        kruskal(*langs_ranks.values())[1] * weights.get(rel, 0)
                        for rel, langs_ranks in rels_langs_ranks.items()
                    ])

                    yield Metric(query_id=qid, measure=measure, value=value)


_provider = PEERProvider()
providers.register(_provider)
ir_measures.DefaultPipeline.providers.append(_provider)


__all__ = ['PEER']