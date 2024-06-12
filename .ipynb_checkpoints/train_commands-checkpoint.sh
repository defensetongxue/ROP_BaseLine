#!/bin/bash

python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-2 --lr 1e-3 
python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-2 --lr 1e-4 
python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-3 --lr 1e-3 
python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-3 --lr 1e-4 
python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-4 --lr 1e-3 
python -u test_stage.py --cfg ./configs/resnet50.json --split_name all --wd 5e-4 --lr 1e-4 
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-2 --lr 1e-3 --resize 299
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-2 --lr 1e-4 --resize 299
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-3 --lr 1e-3 --resize 299
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-3 --lr 1e-4 --resize 299
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-4 --lr 1e-3 --resize 299
python -u test_stage.py --cfg ./configs/inceptionv3.json --split_name all --wd 5e-4 --lr 1e-4 --resize 299
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-2 --lr 1e-3 
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-2 --lr 1e-4 
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-3 --lr 1e-3 
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-3 --lr 1e-4 
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-4 --lr 1e-3 
python -u test_stage.py --cfg ./configs/default.json --split_name all --wd 5e-4 --lr 1e-4 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-2 --lr 1e-3 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-2 --lr 1e-4 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-3 --lr 1e-3 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-3 --lr 1e-4 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-4 --lr 1e-3 
python -u test_stage.py --cfg ./configs/efficientnet_b7.json --split_name all --wd 5e-4 --lr 1e-4 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-2 --lr 1e-3 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-2 --lr 1e-4 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-3 --lr 1e-3 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-3 --lr 1e-4 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-4 --lr 1e-3 
python -u test_stage.py --cfg ./configs/mobelnetv3_small.json --split_name all --wd 5e-4 --lr 1e-4 
