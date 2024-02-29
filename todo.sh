# python -u   train_ridge.py --cfg ./configs/default.json --split_name clr_1 --resize 224
# python -u   train_ridge.py --cfg ./configs/default.json --split_name clr_2 --resize 224
# python -u   train_ridge.py --cfg ./configs/default.json --split_name clr_3 --resize 224
# python -u   train_ridge.py --cfg ./configs/default.json --split_name clr_4 --resize 224

python -u  train_ridge.py --cfg ./configs/efficientnet_b7.json --split_name clr_1 
python -u   train_ridge.py --cfg ./configs/efficientnet_b7.json --split_name clr_2 
python -u   train_ridge.py --cfg ./configs/efficientnet_b7.json --split_name clr_3 
python -u   train_ridge.py --cfg ./configs/efficientnet_b7.json --split_name clr_4 
python -u test_ridge.py  --cfg ./configs/efficientnet_b7.json --split_name clr_1 

python -u   train_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_1 
python -u   train_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_2 
python -u   train_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_3 
python -u   train_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_4 
python -u   test_ridge.py --cfg ./configs/mobelnetv3_small.json --split_name clr_1 

python -u   train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --resize 299
python -u   train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_2 --resize 299
python -u   train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_3 --resize 299
python -u   train_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_4 --resize 299
python -u   test_ridge.py --cfg ./configs/inceptionv3.json --split_name clr_1 --resize 299

python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_2 
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_3 
python -u   train_ridge.py --cfg ./configs/resnet50.json --split_name clr_4 
python -u   test_ridge.py --cfg ./configs/resnet50.json --split_name clr_1 
python ring.py