# Transform and Tell: Entity-Aware News Image Captioning

![Teaser](figures/teaser.png)

## Requirements

```sh
conda env create -f conda.yaml
conda activate tell
python -m ipykernel install --user --name tell --display-name "tell"
cd libs/apex
git submodule init && git submodule update .
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../.. && python setup.py develop
spacy download en_core_web_lg
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

# Object detection for goodnews (takes 19 hours)
CUDA_VISIBLE_DEVICES=0 python scripts/annotate_yolo3.py \
    --source data/goodnews/images \
    --output data/goodnews/objects \
    --dataset goodnews

# Object detection for nytimes (takes 19 hours)
CUDA_VISIBLE_DEVICES=0 python scripts/annotate_yolo3.py \
    --source data/nytimes/images \
    --output data/nytimes/objects \
    --dataset nytimes

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
mongodump --host=localhost --port=27017 --gzip --archive=data/mongobackups/2020-03-05

# Restore database
mongorestore --host=localhost --port=27017 --drop --gzip --archive=data/mongobackups/2020-03-05
```

## Setting Up the Server

```sh
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update && sudo apt full-upgrade -y

sudo apt install -y htop vim zsh tmux ruby-full gcc-7 g++-7 gfortran-7 cmake \
    yarn certbot nginx python-certbot-nginx

# Update current user's password
sudo passwd ubuntu

# Switch to zsh
git clone --recursive https://github.com/sorin-ionescu/prezto.git "${ZDOTDIR:-$HOME}/.zprezto"
chsh -s $(which zsh)
# Log out and log in again, now as zsh. When prompted, select "Quit and do
# nothing" since we want prezto to create its own zsh configuration.
setopt EXTENDED_GLOB
for rcfile in "${ZDOTDIR:-$HOME}"/.zprezto/runcoms/^README.md(.N); do
  ln -s "$rcfile" "${ZDOTDIR:-$HOME}/.${rcfile:t}"
done

sudo gem install tmuxinator
# Get tmux plugin manager
wget https://raw.githubusercontent.com/tmuxinator/tmuxinator/master/completion/tmuxinator.zsh -P ~/.bin
cd ~; git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
# Restore ~/.tmux.conf.local and ~/.tmux/yank.sh from backup
chmod +x ~/.tmux/yank.sh
mkdir ~/.config/tmuxinator
cp /home/ubuntu/projects/transform-and-tell/tmuxinator.yml ~/.config/tmuxinator/tell.yml
# Inside a tmux session, run ` + I to reload the tmux config.

# Install the ultimate .vimrc
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh
# Restore my_configs.vim so that we can paste properly

# Generate key to access GitHub
ssh-keygen -t rsa -b 4096 -C "alasdair.tran@anu.edu.au" -f ~/.ssh/tell_rsa
# Add public key to GitHub

echo "Host *
    IdentityFile ~/.ssh/tell_rsa" > ~/.ssh/config

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 10
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 10
sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-7 10
sudo update-alternatives --set gcc /usr/bin/gcc-7
sudo update-alternatives --set g++ /usr/bin/g++-7
sudo update-alternatives --set gfortran /usr/bin/gfortran-7

# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
zsh Anaconda3-2020.02-Linux-x86_64.sh
rm -rfv Anaconda3-2020.02-Linux-x86_64.sh
source ~/.zshrc
conda update -y conda
conda update -y anaconda

mkdir projects && cd projects
git clone git@github.com:alasdairtran/transform-and-tell.git
cd transform-and-tell
conda env create

sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 'Nginx Full'
sudo ufw enable

sudo certbot certonly --nginx -d transform-and-tell.ml \
    -d www.transform-and-tell.ml \
    -d api.transform-and-tell.ml \
    -d admin.transform-and-tell.ml

sudo rm -rfv ~/lib/nginx /etc/nginx/sites-enabled/default
# Restore /etc/nginx/nginx.conf and /etc/nginx/conf.d/transform-and-tell.conf

# Verify the syntax of our configuration edits.
sudo nginx -t
# Reload Nginx to load the new configuration.
sudo systemctl restart nginx
```
