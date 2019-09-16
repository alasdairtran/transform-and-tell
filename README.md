# newser

News captioning system

## Requirements

```sh
pip install git+https://github.com/salaniz/pycocoevalcap
```

## Getting Data

```sh
# Start local MongoDB server on port 27017
mkdir data/mongodb
mongod --bind_ip_all --dbpath data/mongodb

# Get article URLs from New York Times
python scripts/get_urls.py API_KEY

# Get articles, images, and captions from New York Times
python scripts/get_articles.py

# To reproduce the numbers in Good News, we need to have the same dataset as
# the original paper.
mkdir data/goodnews
wget 'https://drive.google.com/uc?export=download&id=1rl-3DgMRNV8g0AptwKRoYonNkYfT26sf' -O data/goodnews/img_splits.json
wget 'https://drive.google.com/uc?export=download&id=18078qCfdjOHuu75SjBLGNUSiIeq6zxJ-' -O data/goodnews/image_urls.json
URL="https://docs.google.com/uc?export=download&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD" -O data/goodnews/article_caption.json && rm -rf /tmp/cookies.txt
python scripts/get_goodnews.py

python scripts/spacize.py # takes 2-3h
python scripts/annotate_corefs.py # takes about 4 days
python scripts/count_words.py # takes 35m
```

## Training and Evaluation

```sh
CUDA_VISIBLE_DEVICES=0 newser train expt/1_baseline/config.yaml -f

CUDA_VISIBLE_DEVICES=0 newser generate expt/1_baseline/config.yaml -m expt/1_baseline/serialization/best.th

CUDA_VISIBLE_DEVICES=1 newser evaluate expt/1_baseline/config.yaml -m expt/1_baseline/serialization/best.th --overrides '{"validation_iterator": {"batch_size": 8}}'
```
