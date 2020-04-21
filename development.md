# Transform and Tell

News captioning system

## Requirements

```sh
# Install package and dependencies
python setup.py develop

cd $TORCH_HOME
git clone git@github.com:pytorch/fairseq.git
rm -rf pytorch_fairseq_master
mv fairseq pytorch_fairseq_master
cd pytorch_fairseq_master && git checkout 32335404f09c47cccbfbf731abc4c510d0eef043
cd ../.. && rm -rf pytorch_fairseq
```

## Server

```sh
yarn create react-app tatdemo
django-admin startproject backend

cd backend
python manage.py startapp tat
python manage.py migrate
python manage.py runserver
```

## Maintenance

```sh
# Back up database
mongodump --db nytimes --host=localhost --port=27017 --gzip --archive=data/mongobackups/nytimes-2020-04-21
mongodump --db goodnews --host=localhost --port=27017 --gzip --archive=data/mongobackups/goodnews-2020-04-21

# Restore database
mongorestore --db nytimes --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/nytimes-2020-04-21
mongodump --db goodnews --host=localhost --port=27017 --gzip --archive=data/mongobackups/goodnews-2020-04-21

# Archive scraped images
tar -zcf images_processed.tar.gz images_processed
```
