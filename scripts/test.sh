CUDA_VISIBLE_DEVICES=1 python test.py \
	-mode predcls \
	-datasize large \
	-data_path /data0/datasets/ActionGenome/dataset/ag/ \
    -model_path ./checkpoints/debug.tar \
    -pred_contact_threshold 1.0 \
	-window_size 4 \
	-N_layer 1 \
	-enc_layer_num 2 \
	-dec_layer_num 2 \
	-use_spatial_prior \
	-use_temporal_prior \

