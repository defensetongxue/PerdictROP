python -u train.py --cfg ./YAML/inception.yaml
python -u test.py
python -u train.py --cfg ./YAML/vgg.yaml
python -u test.py --cfg ./YAML/vgg.yaml