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
mongodump --host=localhost --port=27017 --gzip --archive=data/mongobackups/2019-10-26

# Restore database
mongorestore --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/2019-10-26
```
