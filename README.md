# Transform and Tell

News captioning system

## Requirements

```sh
# Install package and dependencies
python setup.py develop
```

## Getting Data

```sh
# Start local MongoDB server on port 27017
mkdir data/mongodb
mongod --bind_ip_all --dbpath data/mongodb --wiredTigerCacheSizeGB 10

# Get article URLs from New York Times. Register for an API at
# https://developer.nytimes.com/apis
python scripts/get_urls.py API_KEY
# Construct the NYTimes800k dataset
python scripts/get_articles_nytimes.py
python scripts/process_images.py -i data/nytimes/images -o data/nytimes/images_processed # takes 6h
python scripts/annotate_nytimes.py
python scripts/detect_facenet_nytimes.py

# To reproduce the numbers in Good News, we need to have the same dataset as
# the original paper.
mkdir data/goodnews
wget 'https://drive.google.com/uc?export=download&id=1rl-3DgMRNV8g0AptwKRoYonNkYfT26sf' -O data/goodnews/img_splits.json
wget 'https://drive.google.com/uc?export=download&id=18078qCfdjOHuu75SjBLGNUSiIeq6zxJ-' -O data/goodnews/image_urls.json
URL="https://docs.google.com/uc?export=download&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate $URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rswGdNNfl4HoP9trslP0RUrcmSbg1_RD" -O
data/goodnews/article_caption.json && rm -rf /tmp/cookies.txt
python scripts/get_articles_goodnews.py
python scripts/process_images.py -i data/goodnews/images -o data/goodnews/images_processed
python scripts/annotate_goodnews.py
python scripts/detect_facenet_goodnews.py
# To reproduce the original results, first request the output template captions
# from the authors and put the files in data/goodnews/original_results. Then
# run this command to generate the caption file in our format. In our recollected
# Good News dataset, we're missing 162 image-caption pairs in the test set.
python scripts/goodnews_insert.py --output ./data/goodnews/original_results/with\ article/vis_show_attend_tell_full_TBB.json
python scripts/goodnews_insert.py --output ./data/goodnews/original_results/with\ article/vis_show_attend_tell_full_avg.json
python scripts/goodnews_insert.py --output ./data/goodnews/original_results/with\ article/vis_show_attend_tell_full_wavg.json

# Compute basic statistics on the data
python scripts/compute_name_statistics.py
python scripts/compute_data_statistics.py
python scripts/get_unknown_caption_names.py
```

## Training and Evaluation

```sh
# To train the full model without copying
CUDA_VISIBLE_DEVICES=0 tell train expt/nytimes/7_transformer_faces/config.yaml -f
CUDA_VISIBLE_DEVICES=0 tell evaluate expt/nytimes/7_transformer_faces/config.yaml -m expt/nytimes/7_transformer_faces/serialization/best.th
python scripts/compute_metrics.py -c data/nytimes/name_counters.pkl expt/nytimes/7_transformer_faces/serialization/generations.jsonl

# To train the copying module
CUDA_VISIBLE_DEVICES=0 tell train expt/nytimes/8_transformer_copy/config.yaml -f
CUDA_VISIBLE_DEVICES=0 tell evaluate expt/nytimes/8_transformer_copy/config.yaml -m expt/nytimes/8_transformer_copy/serialization/best.th
python scripts/compute_metrics.py -c data/nytimes/name_counters.pkl expt/nytimes/8_transformer_copy/serialization/generations.jsonl
```

## Maintenance

```sh
# Back up database
mongodump --host=localhost --port=27017 --gzip --archive=data/mongobackups/2019-11-19

# Restore database
mongorestore --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/2019-11-19
```
