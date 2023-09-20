<div align="center">
<h1> OLViT: Multi-Modal State Tracking via Attention-Based Embeddings for Video-Grounded Dialog  </h1>
    
**Anonymous** :ghost: <br>

**Submitted to LREC-COLING 2024** <br>

<img src="misc/teaser.png" width="100%" align="middle"><br><br>

</div>

# Table of Contents
* [Setup and Dependencies](#Setup-and-Dependencies)
* [Download Data](#Download-Data)
* [Training](#Training)
* [Testing](#Testing)
* [Results](#Results)
* [Acknowledgements](#Acknowledgements)

# Setup and Dependencies
We implemented our model using Python 3.7, PyTorch 1.11.0 (CUDA 11.3, CuDNN 8.3.2) and PyTorch Lightning. We recommend to setup a virtual environment using Anaconda. <br>
1. Install [git lfs][1] on your system
2. Clone our repository to download a checpint of our best model and our code
   ```shell
   git lfs install
   git clone this_repo.git
   ```
3. Create a conda environment and install dependencies
   ```shell
   conda create -n olvit python=3.7
   conda activate olvit
   conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
   pip install pytorch-lightning==1.6.3 
   pip install transformers==4.19.2
   pip install torchtext==0.12.0
   pip install wandb nltk pandas 
    ```
# Download Data
1. [DVD][2] and [SIMMC 2.1][3] data are included in this repository and will be downloaded using git lfs  
2. Setup the data by executing
   ```shell
   chmod u+x setup_data.sh
   ./setup_data.sh
    ```
3. This will unpack all the data necessary in ```data/dvd/``` and ```data/simmc/``` 

# Training
We trained our model on 3 Nvidia Tesla V100-32GB GPUs. The default hyperparameters need to be adjusted if your setup differs from ours.
## DVD
1. Adjust the config file for DVD according to your hardware specifications in ```config/dvd.json```
2. Execute
```shell
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --cfg_path config/dvd.json
```
3. Checkpoints will be saved in ```checkpoints/dvd/```

## SIMMC 2.1
1. Adjust the config file for SIMMC 2.1 according to your hardware specifications in ```config/simmc.json```
2. Execute
```shell
CUDA_VISIBLE_DEVICES=0,1,2 python train.py --cfg_path config/simmc.json
```
3. Checkpoints will be saved in ```checkpoints/simmc/```

# Testing
1. Execute
```shell
CUDA_VISIBLE_DEVICES=0 python test.py --ckpt_path <PATH_TO_TRAINED_MODEL> --cfg_path <PATH_TO_CONFIG_OF_TRAINED_MODEL>
```

# Results
Training using the default config and a similar hardware setup as ours will result in the following performance

| Model    | BLUE-1 | BLUE-2 | BLUE-3 | BLUE-4 | METEOR | ROUGE-L | CIDEr |
|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-----:| 
| [Prev. SOTA][6] | 77.8 | 65.4 | 54.9 | 46.8 | 30.8 | 61.9 | 1.336 | 
| MST_MIXER | **78.4** | **65.8** | **55.4** | **46.8** | **31.2** | **62.0** | **1.366**| 


## AVSD-DSTC8
1. Set ```dstc=8``` in the ```ckpt/code/mst_mixer.conf```
2. run
```shell
./generate_parallel_avsd.sh mst_mixer/mixer results_avsd_dstc7_ckpt ckpt/ generate 8
```
3. The responses will be saved in ```output/dstc8/```
4. Executing the [eval_tool][7] of AVSD-DSTC8 using the generated repsonses will output the following metrics

| Model    | BLUE-1 | BLUE-2 | BLUE-3 | BLUE-4 | METEOR | ROUGE-L | CIDEr |
|:--------:|:------:|:------:|:------:|:------:|:------:|:-------:|:-----:| 
| [Prev. SOTA][6] | 76.4 | 64.1 | 53.8 | 45.5 | 30.1 | 61.0 | 1.304 | 
| MST_MIXER | **77.5** | **65.7** | **55.6** | **47.0** | **30.5** | **61.8** | **1.331**|

# Acknowledgements
We thank the authors of [RLM][8] for providing their [code][9] that greatly influenced this work. 



[1]: https://git-lfs.com/
[2]: https://github.com/hudaAlamri/DSTC7-Audio-Visual-Scene-Aware-Dialog-AVSD-Challenge
[3]: https://github.com/dialogtekgeek/DSTC8-AVSD_official
[4]: https://wandb.ai/site
[5]: https://drive.google.com/drive/folders/1SlZTySJAk_2tiMG5F8ivxCfOl_OWwd_Q
[6]: https://aclanthology.org/2022.emnlp-main.280/
[7]: https://drive.google.com/file/d/1EKfPtrNBQ5ciKRl6XggImweGRP84XuPi/view?usp=sharing
[8]: https://arxiv.org/abs/2002.00163
[9]: https://github.com/ictnlp/DSTC8-AVSD
