#!/bin/bash

python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-2 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-2 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-2 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-2 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-2 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-2 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-2 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-2 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-3 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-3 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-3 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-3 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-3 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-3 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-3 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-3 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-4 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-4 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-4 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-4 --lr 1e-3 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 --wd 5e-4 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 --wd 5e-4 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 --wd 5e-4 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 --wd 5e-4 --lr 1e-4 
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-2 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-2 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-2 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-2 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-2 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-2 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-2 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-2 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-3 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-3 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-3 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-3 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-3 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-3 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-3 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-3 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-4 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-4 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-4 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-4 --lr 1e-3 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --wd 5e-4 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --wd 5e-4 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --wd 5e-4 --lr 1e-4 --resize 299
python -u train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --wd 5e-4 --lr 1e-4 --resize 299
