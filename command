














#
python3 train.py --network darknet53 --dataset coco --gpus 1,2 --batch-size 16 -j 16 --log-interval 50 --lr-decay-epoch 55,70 --epochs 85 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 15 --label-smooth --no-wd --coop-cfg 3,3,3 --results-dir result_3
#
python3 train.py --network darknet53 --dataset coco --gpus 3,4 --batch-size 16 -j 16 --log-interval 50 --lr-decay-epoch 55,70 --epochs 85 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 15 --label-smooth --no-wd --coop-cfg 5,5,5 --results-dir result_5
#
python3 train.py --network darknet53 --dataset coco --gpus 5,6 --batch-size 16 -j 16 --log-interval 50 --lr-decay-epoch 55,70 --epochs 85 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 15 --label-smooth --no-wd --coop-cfg 7,7,7 --results-dir result_7
# 3 fit
--network darknet53 --dataset coco --gpus 0,1 --batch-size 16 -j 16 --log-interval 100 --lr-decay-epoch 80,100 --epochs 120 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 20 --label-smooth --no-wd --coop-cfg 3,3,3 --results-dir result_3_fit --fit-epoch 60
# 5 fit
--network darknet53 --dataset coco --gpus 8,9 --batch-size 16 -j 16 --log-interval 100 --lr-decay-epoch 80,100 --epochs 120 --syncbn --warmup-epochs 2 --mixup --no-mixup-epochs 20 --label-smooth --no-wd --coop-cfg 5,5,5 --results-dir result_5_fit --fit-epoch 60

