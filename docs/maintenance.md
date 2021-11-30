# Maintenance

We document some useful code for maintenance. You probably don't need to worry
about this.

```sh
# Back up database
mongodump --db nytimes --host=localhost --port=27017 --gzip --archive=data/mongobackups/nytimes-2020-04-21.gz
mongodump --db goodnews --host=localhost --port=27017 --gzip --archive=data/mongobackups/goodnews-2020-04-21.gz

# Restore database
mongorestore --db nytimes --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/nytimes-2020-04-21.gz
mongorestore --db goodnews --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/goodnews-2020-04-21.gz

# Archive scraped images
tar -zcf data/nytimes/images_processed.tar.gz data/nytimes/images_processed
tar -zcf data/nytimes/images.tar.gz data/nytimes/images
tar -zcf data/goodnews/images_processed.tar.gz data/goodnews/images_processed
tar -zcf data/goodnews/images.tar.gz data/goodnews/images

# Upload data to Nectar Containers
swift upload transform-and-tell expt.7z --info -S 1073741824
swift upload transform-and-tell data --info -S 1073741824
```
