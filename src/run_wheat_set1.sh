datapath=/opt/data1/ECAI/backup/wheat/wheat_set1_train.txt
datasets=('grain')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python main.py \
--gpu 3 \
--seed 0 \
--log_group wheat_set1 \
--log_project GrainAD_Results \
--results_path results \
--run_name run \
net \
-b wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs 25 \
--embedding_size 256 \
--gan_epochs 4 \
--noise_std 0.015 \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin .5 \
--pre_proj 1 \
dataset \
--batch_size 16 \
--resize 224 \
--imagesize 192 "${dataset_flags[@]}" grainset $datapath
