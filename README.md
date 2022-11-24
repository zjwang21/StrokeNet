## Breaking the Representation Bottleneck of Chinese Characters: Neural Machine Translation with Stroke Sequence Modeling (EMNLP 2022)

This repository contains the code for our EMNLP'22 paper [Breaking the Representation Bottleneck of Chinese Characters: Neural Machine Translation with Stroke Sequence Modeling]
## Quick Links

- [Breaking the Representation Bottleneck of Chinese Characters: Neural Machine Translation with Stroke Sequence Modeling](#breaking-the-representation-bottleneck-of-chinese-characters-neural-machine-translation-with-stroke-sequence-modeling)
- [Quick Links](#quick-links)
- [Overview](#overview)
  - [Chinese character to Latinized stroke mapping](#chinese-character-to-latinized-stroke-mapping)
  - [Shared subword vocabulary learning](#shared-subword-vocabulary-learning)
  - [Frequency-aware ciphertext based data augmentation](#frequency-aware-ciphertext-based-data-augmentation)
- [Main Results](#main-results)
- [Train StrokeNet](#train-strokenet)
  - [Requirements](#requirements)
  - [Preprocessing](#preprocessing)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Bugs or Questions?](#bugs-or-questions)
- [Citation](#citation)

## Overview

We propose StrokeNet, a novel technique for Chinese NMT tasks using Latinized stroke sequence of Chinese characters. StrokeNet breaks the representation learning bottleneck and the parameter bottleneck in Chinese NMT tasks, which requires no external data and significantly outperforms several strong prior works. We show that representing Chinese characters in stroke level works well on NMT tasks to bring more internal structure information. We demonstrate that it is possible to implement popular and powerful techniques designed for Latin languages in Chinese NMT tasks. We conduct several analyses on the effects of these Latin language techniques, proving they bring an obvious performance boost in StrokeNet. StrokeNet is a simple and effective approach for Chinese NMT tasks and yields strong results in both high-source and low-source settings. 
### Chinese character to Latinized stroke mapping
A Chinese character is represented by a Latinzied stroke sequence, looking like an English word. The strokes are mapped to lowercased Latin letters by frequency. 
<div align="center">
<img src=figures/LM.png width=90% height=90% />
</div>

### Shared subword vocabulary learning
StrokeNet applies joint vocabulary learning between Latinized stroke sequences and English to capture the rich internal information in Chinese characters and reduce the parameters.
<div align="center">
<img src=figures/subword.png width=90% height=90% />
</div>

### Frequency-aware ciphertext based data augmentation
Replace a character with the k-th character after it by the frequency order to generate ciphertext. Conduct consistency learning between ciphertext and Latinized stroke text.
<div align="center">
<img src=figures/FCDA.png width=90% height=90% />
</div>


## Main Results
We show the main results of StrokeNet on several Chinese NMT tasks. StrokeNet significantly outperforms several strong prior works and achieves SOTA results on the WMT 2017 Zh-En tranlation benchmark without using monolingual data. Meanwhile, StrokeNet decreases the vocabulary size and the parameters obviously. Please see more detailed results in our paper. 

<div align="center">
<img src=figures/main_result.png width=95% height=95% />
</div>


## Train StrokeNet

In the following section, we provide instructions on training StrokeNet with our code.

### Requirements

First, install PyTorch by following the instructions from [the official website](https://pytorch.org/). To faithfully reproduce our results, please use the correct `1.10.1` version corresponding to your platforms/CUDA versions. PyTorch version higher than `1.10.1` should also work. For example, if you use Linux and **CUDA11** ([how to check CUDA version](https://varhowto.com/check-cuda-version/)), install PyTorch by the following command,

```
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Then try runing the following script to install other dependencies.

```bash
pip install -r requirements.txt
cd fairseq-cipherdaug
pip install --editable ./
```

### Preprocessing
#### Data
All example scripts are based on the NIST Zh-En.
All the bash scripts are sufficiently annotated for reference.

Prepare the NIST Zh-En train and evaluation(MT02, MT03, MT04, MT08, MT06) data from https://catalog.ldc.upenn.edu/. 

Use MT06 as the valid data. Place train, valid, test data like this:
```
|-- StrokeNet/data/NIST/
    |-- source/
        |-- train.zh-en.zh
        |-- train.zh-en.en
        |-- valid.zh-en.zh
        |-- valid.zh-en.en
    |-- dev_test/
        | --nist02/
            | --nist02.en
            | --nist02.en1
            | --nist02.en2
            | --nist02.en3
            | --nist02.zh
        | --nist03/
        ......
```

Follow the procedure below to prerpocess the data.

#### Convert Chinese to Latinized stroke sequence and cipher with keys.

```bash
bash $LOC/StrokeNet/scripts/preprocess.sh
```
This creates all parallel Latinized stroke data of cipher-1 and cipher-2 in the output dir.
If you need to generate Latinized stroke data of your own, your file names should follow the rules mentioned in [DATA](#data) and take the following commands:
```bash
python $LOC/StrokeNet/fairseq-cipherdaug/strokenet/zh2letter.py \
    -i input_file_path \
    -o output_file_path \
    -v vocab_file_path \
    --workers n
```
This generates stroke sequence corpus in output_file_path of the files in input_file_path.
```bash
python $LOC/StrokeNet/fairseq-cipherdaug/strokenet/cipher.py \
    -i input_file_path \
    -s zh -t en --workers n \
    --keys 1 2
```
This generates ciphertexts with keys (1 and 2) in input_file_path.


#### Conduct BPE algorithm and binarize the data.
We use subword-nmt for BPE oprations.
For learning and applying BPE algorithm on all relevant files at once, use the `bpe.sh`
```
bash /home/StrokeNet/scripts/bpe.sh
```
Number of BPE merge operations can be changed in bash file.
This part could last for minutes, wait patiently for it to finish.

Then use `multi_binarize.sh` to generate joint multilingual dictionary and binary files for fairseq to use.
```
bash /home/StrokeNet/scripts/multi_binarize.sh
```

### Training

`train.sh` comes loaded with all relevant details to set hyperparameters and start training 
```
bash /home/StrokeNet/scripts/train.sh
```
Part of the key parameters:
```
fairseq-train $DATABIN --save-dir ${CKPT} \
    --lang-dict "${LANG_LIST}" --lang-pairs "${LANG_PAIRS}" \
    --eval-lang-pairs ${EVAL_LANG_PAIRS} \
    --task ${TASK} \                                         
    --arch transformer --share-all-embeddings \                 # Weight tying
    --criterion ${LOSS} --label-smoothing 0.1 \            
    --valid-subset valid --ignore-unused-valid-subsets --batch-size-valid 200 \
```
For keys 1 and 2:  

* `--lang-pairs` should be "zh-en,zh1-en,zh2-en". 
* ` --eval-lang-pairs` shoule be "zh-en,". 
* `--lang-dict` should be a file containing "zh, zh1, zh2, en".  
* `--task` should be "translation_multi_simple_epoch_cipher --prime-src zh --prime-tgt en".  
* `--criterion` should be "label_smoothed_cross_entropy_js --js-alpha 5 --js-warmup 500".   
*  `--js-alpha` is the coefficient of the consistent loss, StrokeNet does [consistency learning](#frequency-aware-ciphertext-based-data-augmentation)

See more details about the hyperparameters in our paper.


### Evaluation

```
bash /home/StrokeNet/scripts/eval.sh
```
Evaluation will be conducted on MT02, MT03, MT04, MT08, ALL OF THEM AND MT06(VALID SET). Results will be generated in the output checkpoint dir.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Zhijun Wang (wzhijun21@gmail.com). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you use StrokeNet in your work:
```
@inproceedings{wang2022StrokeNet,
 author = {Wang, Zhijun and Liu, Xuebo and Zhang, Min},
 booktitle = {Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
 title = {Breaking the Representation Bottleneck of Chinese Characters: Neural Machine Translation with Stroke Sequence Modeling},
 url = {https://arxiv.org/abs/2211.12781}, 
 year = {2022}
}
```
