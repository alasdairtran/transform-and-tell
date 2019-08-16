# newser

News captioning system

## Getting Data

```sh
# Start local MongoDB server on port 27017
mkdir data/mongodb
mongod --bind_ip_all --dbpath data/mongodb

# Get article URLs from New York Times
python scripts/get_urls.py API_KEY

# Get articles, images, and captions from New York Times
python scripts/get_articles.py
```

## Training

```sh
CUDA_VISIBLE_DEVICES=0 newser train expt/1_baseline/config.yaml -f
```
