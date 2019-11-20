# Transform and Tell

News captioning system

## Requirements

```sh
# Install package and dependencies
python setup.py develop
```

## Server

```sh
yarn create react-app tatdemo

```

## Maintenance

```sh
# Back up database
mongodump --host=localhost --port=27017 --gzip --archive=data/mongobackups/2019-10-26

# Restore database
mongorestore --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/2019-10-26
```
