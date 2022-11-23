<!-- <p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p> -->

--------------------------------------------------------------------------------

<!-- Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. -->

This is a fork of version `1.0.0a0+e3fafbd` (Sept/Oct 2021) of [Fairseq](https://github.com/pytorch/fairseq) that has been adapted for the methods and experiments in the `CipherDAug` paper.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

# Changes/Additions for CipherDAug

These are the main changes to Fairseq

* tasks - supports [switchout](fairseq/data/switchout.py) family
  * [multilingual translation with evaluation](fairseq/tasks/translation_multi_simple_epoch_eval.py)
  * [multilingual translation with cipherdaug](fairseq/tasks/translation_multi_simple_epoch_cipher.py) extends the task above :arrow_up:

* data - the tasks depend on these
  * [language triple dataset](fairseq/data/language_triple_dataset.py) -- very much like `language pair dataset` for the classic translation task, but supports multisource [source1 + source2]:arrow_right:target
  * improved general purpose [multilingual data manager](fairseq/data/multilingual/multilingual_data_manager_w_eval.py) that supports evaluation and language pair/triple datasets necessary for multisource

* criterions - loss functions
  * [symmetric KL loss](fairseq/criterions/label_smoothed_cross_entropy_js.py) can be easily changed to jensen-shannon divergence as well

There might be a few tiny modifications here and there that are not listed here but can be easily traced through a code walkthrough.

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
