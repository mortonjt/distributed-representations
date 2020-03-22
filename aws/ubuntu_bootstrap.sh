# install virtualenv and other tools
sudo apt install virtualenv
virtualenv roberta -p python3.7
source /home/ubuntu/roberta/bin/activate
pip install torch=1.4
pip install git+git@github.com:mortonjt/fairseq.git
