# Probability of Equal Expected Rank

This package is the Python implementation of the MLIR fairness measure 
"Probability of Equal Expected Rank" using `ir_measures`. 

## How to use it

You can either directly install it from PyPi through
```bash
pip install peer_measure
```

Or install the GitHub version
```bash
pip install pip@git+https://github.com/hltcoe/peer_measure
```

When importing, please import both `peer_measure` and `ir_measures`. 
```python
from peer_measure import PEER
import ir_measures
```

Please refer to the documentation of `ir_measures` for the general usage. 

## Parameters

`PEER` takes two required parameters: `weights` and `lang_mapping`. 
- `weights`: a int-to-float dictionary specifying the weight for each relevance level. The weight have be sum up to 1.0. 
- `lang_mapping`: a str-to-str dictionary with keys being the `doc_id` and values being the language id of the correspoding document. 

You can specify these parameters and the rank cutoff when declaring the measure instance. For example,
```python
measure = PEER(weights={0: 0, 1: 0.5, 2:0, 3: 0.5}, lang_mapping=...)@20
```

Please refer to our paper for detail definition and implication of the parameters. 

## Citation

Please consider citing our paper if you use this measure. 

```bibtex
@inproceedings{peer,
	author = {Eugene Yang and Thomas Jänich and James Mayfield and Dawn Lawrie},
	title = {Language Fairness in Multilingual Information Retrieval},
	booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) (Short Paper) (Accepted)},
	year = {2024}, 
	doi = {10.1145/3626772.3657943}
}
```