This Repository is my assignment for cs3324 on IQA model design, based on [chaofengc/IQA-PyTorch: 👁️ 🖼️ 🔥PyTorch Toolbox for Image Quality Assessment, including LPIPS, FID, NIQE, NRQM(Ma), MUSIQ, NIMA, DBCNN, WaDIQaM, BRISQUE, PI and more... (github.com)](https://github.com/chaofengc/IQA-PyTorch)

## Quick Start:

```
# Install the requirements, a python 3.8 environment is recommended.
pip install -r requirements.txt

# As this repository is basically my modified version of pyiqa, you need to first uninstall the original pyiqa.
pip unintall pyiqa

# Then install my version.
python setup.py develop
```

**Install torch form [PyTorch](https://pytorch.org/) if you want to use cuda.**

## Extract the AGIQA-3K dataset to the './datasets/' directory.:

Now the structure should be like:

```
your_cloned_repo/
|
|___ datasets/
|    |
|    |___ AGIQA-3K/
|    |    |
|    |    |___ image1.jpg ...
|    |
|    |___ other_datasets/
|    |
|    |___ meta_info/
|         |
|         |___ AGIQA-3K.pkl
|         |
|         |___ meta_info_AGIQA-3K.csv
|
|___ other_dirs/

```

## Train my SVD-CNNIQA:

You can directly click on './start.bat', if you are using Windows and exactly one GPU. 

Or run by command:

```
# In the env you set up previously:
python pyiqa/train.py -opt options/train/train_AGICQ-3K_myiqa.yml

```

If the training process fails, try to adjust the parameters 'num_worker_per_gpu' and 'batch_size_per_gpu'  in the settings file '.\options\train\train_AGICQ-3K_myiqa.yml'.

## My Results:

Original results stored in  './myresults', including logs and '.pth' file of SVD_CNNIQA.
