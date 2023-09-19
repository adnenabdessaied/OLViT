<div align="center">
<h1> OLViT: Multi-Modal State Tracking via Attention-Based Embeddings for Video-Grounded Dialog  </h1>
    
**Anonymous** :ghost: <br>

**Submitted to LREC-COLING 2024** <br>

<img src="misc/teaser.png" width="100%" align="middle"><br><br>

</div>

# Table of Contents
* [Setup and Dependencies](#Setup-and-Dependencies)
* [Download Data](#Download-Data)
* [Pre-trained Checkpoint](#Pre-trained-Checkpoint)
* [Training](#Training)
* [Response Generation](#Response-Generation)
* [Results](#Results)
* [Acknowledgements](#Acknowledgements)

# Setup and Dependencies
We implemented our model using Python 3.7 and PyTorch 1.12.0 (CUDA 11.3, CuDNN 8.3.2). We recommend to setup a virtual environment using Anaconda. <br>
1. Install [git lfs][1] on your system
2. Clone our repository to download a checpint of our best model and our code
   ```shell
       git lfs install
       git clone this_repo.git
   ```
3. Create a conda environment and install dependencies
   ```shell
       conda create -n mst_mixer python=3.7
       conda activate mst_mixer
       conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
       conda install pyg -c pyg
       conda install pytorch-scatter -c pyg  # pytorch >= 1.8.0
       conda install -c huggingface transformers
       pip intall wandb glog pyhocon 
    ```
# Download Data
1. Download the [AVSD-DSTC7][2] and [AVSD-DSTC8][3] data
2. Place the raw json files in ```raw_data/``` and the features in ```features/```
3. Prepeocess and save the input features for faster training as indicated in ```custom_datasets/```

# Pre-trained Checkpoint
We provide a checkpoint of our best model in the ```ckpt/``` folder.

# Training
We trained our model on 8 Nvidia Tesla V100-32GB GPUs. The default hyperparameters in ```config/mst_mixer.conf``` need to be adjusted if your setup differs from ours.
```shell
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py \
         --mode train \ 
         --tag unique_training_tag \
         --wandb_mode online \
         --wandb_project your_wandb_project_name
```
To deactivate [wandb][4] logging, use ```--wandb_mode disabled```.
On a similar setup to ours, this will take roughly 20h to complete.

# Response Generation
The inference runtime of our model takes circa 2s to generate one response. Since each of AVSD-DSTC7 and AVSD-DSTC8 have 1,710 questions each, inference using a single model will take almost 1h. Thus, we provide a shell script that runs 16 models on parallel (2x model/GPU) to reduce the runtime to only 6mn.
## AVSD-DSTC7
1. Set ```dstc=7``` in the ```.conf``` file of your trained networks. in The default setting, can find this under ```logs/unique_training_tag/code/config/mst_mixer.conf``` 
2. Generate the responses
```shell
./generate_parallel_avsd.sh mst_mixer/mixer results_avsd_dstc7 logs/unique_training_tag generate 7
```
3. All responses will be saved in ```output/dstc7/```
## AVSD-DSTC8
1. Set ```dstc=8``` in the ```.conf``` file of your trained networks. in The default setting, can find this under ```logs/unique_training_tag/code/config/mst_mixer.conf``` 
2. Generate the responses
```shell
./generate_parallel_avsd.sh mst_mixer/mixer results_avsd_dstc8 logs/unique_training_tag generate 8
```
3. All responses will be saved in ```output/dstc8/```

# Results
To evaluate our best model on 
## AVSD-DSTC7
1. Set ```dstc=7``` in the ```ckpt/code/mst_mixer.conf```
2. run
```shell
   ./generate_parallel_avsd.sh mst_mixer/mixer results_avsd_dstc7_ckpt ckpt/ generate 7
```
3. The responses will be saved in ```output/dstc7/```
4. Executing the [eval_tool][5] of AVSD-DSTC7 using the generated repsonses will output the following metrics

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
