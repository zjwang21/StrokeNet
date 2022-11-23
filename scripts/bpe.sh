#!/bin/bash
LOC=/  #set your local directory
DATA=$LOC/StrokeNet/data/NIST/strokenet_data  #set your data root dir
zhx_en_2keys() {
    SRCS=(
        "zh"
        "zh1"
        "zh2"
    )

    TGTS=(
        "en"
        "zh"
    )
}

##############################
#### call the config here ####

zhx_en_2keys


mkdir -p $DATA/bpe
#learning joint bpe in all training data.
cat $DATA/train.${SRCS}-${TGTS}.${SRCS} $DATA/train.${SRCS[1]}-${TGTS}.${SRCS[1]} \
$DATA/train.${SRCS[2]}-${TGTS}.${SRCS[2]} $DATA/train.${SRCS}-${TGTS}.${TGTS} > $DATA/train.all

echo "Learning joint bpe......"
subword-nmt learn-joint-bpe-and-vocab --input $DATA/train.all -s 30000  \
-o $DATA/bpe/joint.code --min-frequency 50 --write-vocabulary $DATA/bpe/joint.vocab

#apply bpe
echo "Applying bpe......"
for SPLIT in train valid; do
    for SRC in ${SRCS[@]}; do
        for TGT in ${TGTS[@]}; do
            if [ ! ${SRC} = ${TGT} ]; then
                echo "Generate $SPLIT.bpe.$SRC-$TGT.$SRC"
                subword-nmt apply-bpe -c $DATA/bpe/joint.code < $DATA/$SPLIT.$SRC-$TGT.$SRC > $DATA/bpe/$SPLIT.bpe.$SRC-$TGT.$SRC
                echo "Generate $SPLIT.bpe.$SRC-$TGT.$TGT"
                subword-nmt apply-bpe -c $DATA/bpe/joint.code < $DATA/$SPLIT.$SRC-$TGT.$TGT > $DATA/bpe/$SPLIT.bpe.$SRC-$TGT.$TGT
            fi
        done
    done
done