
# 参数含义
# https://github.com/horovod/horovod/blob/master/docs/mpirun.rst

export BERT_BASE_DIR=/data/models/bert/uncased_L-2_H-128_A-2
export GLUE_DIR=/data/datasets/glue/glue_data/


mpirun -np 1 \
    -H localhost:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_classifier_hvd.py \
    --task_name=MRPC \
    --do_train=true \
    --do_eval=true \
    --data_dir=$GLUE_DIR/MRPC \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=4.0 \
    --output_dir=/data/tmp/mrpc_output/