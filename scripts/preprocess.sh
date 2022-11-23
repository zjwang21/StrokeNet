#!/bin/bash
LOC=/  #set your local directory
DATA=$LOC/StrokeNet/data/NIST
TOOL=$LOC/StrokeNet/fairseq-cipherdaug/strokenet

SRC=zh  #set source language name
TGT=en  #set target language name

#create strokenet data dir, the final data prepared for bpe algorithm will all be here
mkdir -p $DATA/strokenet_data

for SPLIT in train valid
do
#generating latinized stroke Chinese corpus
python $TOOL/zh2letter.py -i $DATA/source/$SPLIT.$SRC-$TGT.$SRC \
    -o $DATA/strokenet_data/$SPLIT.$SRC-$TGT.$SRC -v $LOC/StrokeNet/vocab/zh2letter.txt --workers 5

cp $DATA/source/$SPLIT.$SRC-$TGT.$TGT $DATA/strokenet_data/$SPLIT.$SRC-$TGT.$TGT
done

#generating cipher-text data
python $TOOL/cipher.py -i $DATA/strokenet_data -s $SRC -t $TGT --workers 5 --keys 1 2