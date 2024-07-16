conda create --name andrea python=3.8
conda activate andrea
conda install -c anaconda cudatoolkit -y
conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install pytorch torchvision torchaudio -c pytorch -c nvidia -y
sudo apt install -y build-essential cmake python3-dev
pip3 install plotly pandas numpy nbformat scikit-learn pybind11 cupy-cuda12x
conda install openpyxl
conda install -c conda-forge torchview
conda install -c rapidsai -c nvidia -c conda-forge cuml cudf