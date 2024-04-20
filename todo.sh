python -u   train_ridge.py --cfg ./configs/default.json --split_name all --resize 224

python -u  train_ridge.py --cfg ./configs/efficientnet_b7.json --split_name clr_1 

python -u   train_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_1

python -u   train_ridge.py --cfg ./configs/inceptionv3.json --split_name all --resize 299

python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-2 --lr 1e-4
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-2 --lr 1e-4
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-2 --lr 1e-4
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-2 --lr 1e-4