# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


example console command for train.py:
python aipnd-project/train.py --data_dir flowers --model resnet18 --depth 2 --learning_rate 0.01 --dropout 0.2 --epochs 5 --stop_num 10 --device gpu --model_name resnetcheckpoint


example console command for predict.py:
python aipnd-project/predict.py --image_path flowers/test/80/image_02020.jpg --model checkpoint.pth --device gpu

