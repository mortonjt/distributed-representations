# install virtualenv and other tools
sudo yum install python-virtualenv
virtualenv roberta -p python3.6
source /home/ec2-user/roberta/bin/activate
# TODO: what about GPU drivers and NCCL??
pip install torch==1.4
# TODO: the gcc version is grossly out of date (it should be version 8.3 not 4.8)
# TODO: put exact version numbers of all of the packages
pip install numpy scipy
pip install tensorboardX
pip install git+https://github.com/pytorch/fairseq.git
