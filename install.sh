cd src

conda install "setuptools <65"

pip install numpy==1.23.5

MAX_JOBS=8 python setup.py install