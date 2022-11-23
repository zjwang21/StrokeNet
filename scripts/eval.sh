LOC=/ #set your local dir 
OUTPUT=$LOC/StrokeNet/checkpoints/NIST  # add your experiment name
DEV=$LOC/StrokeNet/data/NIST/dev_test/
ROOT=$LOC/StrokeNet/data/NIST/strokenet_data/bpe
for test in 02 03 04 06 08 09
do
RESULT=$OUTPUT/test.$test            

python $LOC/StrokeNet/preprocess/zh2letter.py -i $DEV/nist$test/nist$test.zh \
-o $DEV/nist$test/nist$test.stroke.zh -v $LOC/StrokeNet/vocab/zh2letter.txt --workers 1

subword-nmt apply-bpe -c $ROOT/joint.code \
< $DEV/nist$test/nist$test.stroke.zh \
> $DEV/nist$test/nist$test.bpe.zh

subword-nmt apply-bpe -c $ROOT/joint.code \
< $DEV/nist$test/nist$test.en \
> $DEV/nist$test/nist$test.bpe.en

fairseq-preprocess --testpref $DEV/nist$test/nist$test.bpe \
-s zh -t en \
--srcdict $ROOT/bin/jointdict.txt \
--tgtdict $ROOT/bin/jointdict.txt \
--destdir $OUTPUT/nist$test-bin

for ck in _best
do
fairseq-generate $OUTPUT/nist$test-bin --task translation_multi_simple_epoch \
    --lang-tok-style "multilingual" --source-lang zh --target-lang en --encoder-langtok "tgt" \
    --lang-dict "$ROOT/bin/langs.file" \
    --lang-pairs "zh-en" \
    --path $OUTPUT/checkpoint$ck.pt \
    --batch-size 128 --beam 5 --remove-bpe > ${RESULT}.nist$test.$ck.all

cat ${RESULT}.nist$test.$ck.all | grep -P "^H" |sort -V |cut -f 3- > ${RESULT}.nist$test.$ck.hyp

sacrebleu -tok 'none' -s 'none' \
$DEV/nist$test/nist$test.en \
$DEV/nist$test/nist$test.en1 \
$DEV/nist$test/nist$test.en2 \
$DEV/nist$test/nist$test.en3 \
< ${RESULT}.nist$test.$ck.hyp > ${RESULT}.nist$test.$ck.score
done
done