# install virtualenv and other tools
sudo yum install python-virtualenv
virtualenv roberta -p python3.6
source /home/ec2-user/roberta/bin/activate
# TODO: what about GPU drivers and NCCL??
pip install torch==1.4
pip install git+git@github.com:mortonjt/fairseq.git
