python -u  train.py --cfg ./configs/efficientnet_b7.json --split_name 1 
python -u  train.py --cfg ./configs/efficientnet_b7.json --split_name 2 
python -u  train.py --cfg ./configs/efficientnet_b7.json --split_name 3 
python -u  train.py --cfg ./configs/efficientnet_b7.json --split_name 4 

python -u  train.py --cfg ./configs/resnet18.json --split_name 1 
python -u  train.py --cfg ./configs/resnet18.json --split_name 2 
python -u  train.py --cfg ./configs/resnet18.json --split_name 3 
python -u  train.py --cfg ./configs/resnet18.json --split_name 4 

python -u  train.py --cfg ./configs/mobelnetv3_small.json --split_name 1 
python -u  train.py --cfg ./configs/mobelnetv3_small.json --split_name 2 
python -u  train.py --cfg ./configs/mobelnetv3_small.json --split_name 3 
python -u  train.py --cfg ./configs/mobelnetv3_small.json --split_name 4 

python -u  train.py --cfg ./configs/inceptionv3.json --split_name 1 --resize 299
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 2 --resize 299
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 3 --resize 299
python -u  train.py --cfg ./configs/inceptionv3.json --split_name 4 --resize 299


python -u  train.py --cfg ./configs/resnet50.json --split_name 1 
python -u  train.py --cfg ./configs/resnet50.json --split_name 2 
python -u  train.py --cfg ./configs/resnet50.json --split_name 3 
python -u  train.py --cfg ./configs/resnet50.json --split_name 4 

python -u  train.py --cfg ./configs/vgg16.json --split_name 1 
python -u  train.py --cfg ./configs/vgg16.json --split_name 2 
python -u  train.py --cfg ./configs/vgg16.json --split_name 3 
python -u  train.py --cfg ./configs/vgg16.json --split_name 4 

python -u  train.py --cfg ./configs/mobelnetv3_large.json --split_name 1 
python -u  train.py --cfg ./configs/mobelnetv3_large.json --split_name 2 
python -u  train.py --cfg ./configs/mobelnetv3_large.json --split_name 3 
python -u  train.py --cfg ./configs/mobelnetv3_large.json --split_name 4 

python ring.py
shutdown