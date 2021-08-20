# Setting Up Demo

You probably don't need to read this unless you want to set up the demo
on your own machine. These instructions contain some stuff that is specific
to our own server.

## Quick Start

```sh
# Start the frontend
conda activate tell
python setup.py develop
cd demo/frontend
yarn
rm -rf build && yarn build && yarn serve -s build

# Restore model weight
rsync -rlptzhe ssh --info=progress2 best.th tell:~/projects/transform-and-tell/expt/nytimes/8_transformer_faces/serialization/
cd demo/backend
CUDA_VISIBLE_DEVICES=2 python -m tell.server caption

# Run django backend
gunicorn backend.wsgi
```

## Intializing the Demo Project From Scratch

```sh
yarn create react-app tatdemo
django-admin startproject backend

cd backend
python manage.py startapp tat
python manage.py migrate
SECRET_KEY='' DJANGO_DEBUG='False' python manage.py runserver
```

## Setting Up the Demo Server From Scratch

```sh
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt-key adv --refresh-keys --keyserver keyserver.ubuntu.com
sudo apt update && sudo apt full-upgrade -y

sudo apt install -y htop vim zsh tmux ruby-full gcc-7 g++-7 gfortran-7 cmake \
    yarn certbot nginx python-certbot-nginx autossh libgl1

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
# Log out and log in again.

sudo gem install tmuxinator
# Get tmux plugin manager
wget https://raw.githubusercontent.com/tmuxinator/tmuxinator/master/completion/tmuxinator.zsh -P ~/.bin
cd ~; git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
# Install the ultimate .vimrc
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh

# Restore ~/.tmux.conf.local and ~/.tmux/yank.sh from backup
# Restore my_configs.vim so that we can paste properly
rsync -rlptzhe ssh --info=progress2 .vim_runtime/my_configs.vim tell:~/.vim_runtime/
rsync -rlptzhe ssh --info=progress2 .tmux.conf.local tell:~/
rsync -rlptzhe ssh --info=progress2 .tmux/yank.sh tell:~/.tmux/
chmod +x ~/.tmux/yank.sh
mkdir ~/.config/tmuxinator
# Restore ssh config

# Generate key to access GitHub. Update the email accordingly.
ssh-keygen -t rsa -b 4096 -C "first.last@anu.edu.au" -f ~/.ssh/tell_rsa
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
sudo ufw allow ssh comment "SSH"
sudo ufw enable

sudo certbot certonly --nginx -d transform-and-tell.ml \
    -d www.transform-and-tell.ml \
    -d api.transform-and-tell.ml \
    -d admin.transform-and-tell.ml

# Restore nginx configs
rsync -rlptzhe ssh --info=progress2 nginx tell:~/
ssh tell
sudo mv nginx/nginx.conf /etc/nginx/nginx.conf
sudo mv nginx/transform-and-tell.conf /etc/nginx/conf.d/transform-and-tell.conf
sudo rm -rfv ~/lib/nginx /etc/nginx/sites-enabled/default
# Verify the syntax of our configuration edits.
sudo nginx -t
# Reload Nginx to load the new configuration.
sudo systemctl restart nginx

cp /home/ubuntu/projects/transform-and-tell/tmuxinator.yml ~/.config/tmuxinator/tell.yml
# Inside a tmux session, run ` + I to reload the tmux config.
```
