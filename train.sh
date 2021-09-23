export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=19090 sh tools/dist_train.sh configs/ReDet_trans_drone/retinanet_obb_r50_fpn_2x_TD_3cat_wide.py 4
PORT=19190 sh tools/dist_train.sh configs/ReDet_trans_drone/retinanet_obb_r50_fpn_2x_TD_3cat_mix.py 4