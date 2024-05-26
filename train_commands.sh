#!/bin/bash

python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-2 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-2 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-2 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-2 --lr 1e-3
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-2 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-2 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-2 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-2 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-2 --lr 1e-4
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-2 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-3 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-3 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-3 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-3 --lr 1e-3
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-3 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-3 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-3 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-3 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-3 --lr 1e-4
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-3 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-4 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-4 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-4 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-4 --lr 1e-3
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-4 --lr 1e-3
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_1 --wd 5e-4 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_2 --wd 5e-4 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_3 --wd 5e-4 --lr 1e-4
python -u train_ridge.py --cfg ./configs/vgg16.json --split_name clr_4 --wd 5e-4 --lr 1e-4
python -u test_ridge.py --cfg ./configs/vgg16.json --split_name all --wd 5e-4 --lr 1e-4
python ring.py