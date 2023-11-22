python train.py --cfg ./configs/efficientnet_b7.json --split_name 1 
python train.py --cfg ./configs/efficientnet_b7.json --split_name 2 
python train.py --cfg ./configs/efficientnet_b7.json --split_name 3 
python train.py --cfg ./configs/efficientnet_b7.json --split_name 4 

python train.py --cfg ./configs/resnet18.json --split_name 1 
python train.py --cfg ./configs/resnet18.json --split_name 2 
python train.py --cfg ./configs/resnet18.json --split_name 3 
python train.py --cfg ./configs/resnet18.json --split_name 4 

python train.py --cfg ./configs/mobelnetv3_small.json --split_name 1 
python train.py --cfg ./configs/mobelnetv3_small.json --split_name 2 
python train.py --cfg ./configs/mobelnetv3_small.json --split_name 3 
python train.py --cfg ./configs/mobelnetv3_small.json --split_name 4 

python train.py --cfg ./configs/inceptionv3.json --split_name 1 --resize 299
python train.py --cfg ./configs/inceptionv3.json --split_name 2 --resize 299
python train.py --cfg ./configs/inceptionv3.json --split_name 3 --resize 299
python train.py --cfg ./configs/inceptionv3.json --split_name 4 --resize 299
