# 1. install miniconda
# 2. install libraries
conda create --name machining
conda activate machining
conda install pytorch torchvision cpuonly -c pytorch -y
conda install pandas -y
conda install numpy -y
conda install scikit-learn -y
conda install -c anaconda seaborn -y
pip install nptdms
pip install tqdm
pip install xlrd
conda install -c plotly plotly -y
# 3. download project
# 4. install visual studio code