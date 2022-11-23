LOC=/ # set your root project location
ROOT="${LOC}/StrokeNet"
DATAROOT="${ROOT}/data/NIST/strokenet_data/bpe" # set your data root

DATABIN="${DATAROOT}/bin"
mkdir -p $DATABIN

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


##############################
# best if left untouched

DICT=jointdict.txt
echo "Generating joined dictionary for all languages based on BPE.."
# strip the first three special tokens and append fake counts for each vocabulary
cat $DATAROOT/joint.vocab | cut -d ' ' -f1 | sed 's/$/ 100/g' > "$DATABIN/$DICT"

echo "binarizing pairwise langs .."
for SRC in ${SRCS[@]}; do
    for TGT in ${TGTS[@]}; do
        if [ ! ${SRC} = ${TGT} ]; then
            echo "binarizing data ${SRC}-${TGT} data.."
            fairseq-preprocess --source-lang ${SRC} --target-lang ${TGT} \
                --destdir "${DATABIN}" \
                --trainpref "${DATAROOT}/train.bpe.${SRC}-${TGT}" \
                --validpref "${DATAROOT}/valid.bpe.${SRC}-${TGT}" \
                --srcdict "${DATABIN}/${DICT}" --tgtdict "${DATABIN}/${DICT}" \
                --workers 10
        fi
    done
done

echo ""
echo "Creating langs file based on binarised dicts .."
echo "${SRCS}
${SRCS[1]} 
${SRCS[2]}
${TGTS}" > "${DATABIN}/langs.file"
echo "--> ${DATABIN}/langs.file"

