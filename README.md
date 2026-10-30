# Rewriting Conversational Utterances with Instructed LLMs

Code for the WI-IAT 2023 paper
**"Rewriting Conversational Utterances with Instructed Large Language Models"**
(E. Galimzhanova, C. I. Muntean, F. M. Nardini, R. Perego, G. Rocchietti — ISTI-CNR, Pisa).

[Paper (IEEE Xplore)](https://doi.org/10.1109/WI-IAT59888.2023.00014) · [arXiv:2410.07797](https://arxiv.org/abs/2410.07797)

## Overview

We study whether instructed LLMs (ChatGPT, `gpt-3.5-turbo`) can rewrite conversational utterances
into self-contained queries, and which prompts produce the rewrites that lead to the best retrieval.
On TREC CAsT, rewriting with instructed LLMs improves over state-of-the-art rewriting techniques by
up to 25.2% in MRR, 31.7% in Precision@1, 27% in NDCG@3 and 11.5% in Recall@500.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `chat-gpt_call.py` | Rewrites the CAsT utterances with `gpt-3.5-turbo` under different prompts (with or without previous utterances and previous rewrites). |
| 2 | `clean.py` | Removes boilerplate from the model output (e.g. echoed instructions) and truncates the rewrites. |
| 3 | `retrieve_script.py` | First-stage retrieval with PyTerrier (DPH) and evaluation against the qrels. |
| 4 | `reranking_script.py` | Reranks the retrieved passages with MonoT5 (`castorini/monot5-base-msmarco`). |
| 5 | `stat_sig.py` | Paired t-tests against a chosen baseline (PyTerrier, Bonferroni correction) and paper tables. |

`trec/` contains the TREC CAsT 2019–2021 topics, manual rewrites, qrels and baseline runs.
The passage collection and the index are not included.

## Usage

Requirements: Python 3.8+, `openai`, `python-terrier`, `pyterrier_t5`, `transformers`, `pandas`.
Set your OpenAI key through an environment variable rather than in the code.

```bash
python clean.py --input_folder rewritings/raw/ --output_folder rewritings/cleaned/ --year 2019
python retrieve_script.py --trec_path trec/ --rewriting_path rewritings/cleaned/ \
    --year 2019 --index_path <CAsT index> --outpath results/
python reranking_script.py --collection <CASTcollection.tsv> --rewriting_path rewritings/cleaned/ \
    --ranked_path results/for_reranking/ --outpath results/ --year 2019
python stat_sig.py --rewriting_path rewritings/cleaned/ --ranked_path results/reranked/ \
    --outpath results/ --year 2019 --baseline cqr
```

## Citation

```bibtex
@inproceedings{galimzhanova2023rewriting,
  author    = {Galimzhanova, Elnara and Muntean, Cristina Ioana and Nardini, Franco Maria and Perego, Raffaele and Rocchietti, Guido},
  title     = {Rewriting Conversational Utterances with Instructed Large Language Models},
  booktitle = {2023 IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT)},
  pages     = {56--63},
  year      = {2023},
  doi       = {10.1109/WI-IAT59888.2023.00014}
}
```

## License

MIT — see [LICENSE](LICENSE).
