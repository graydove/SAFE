# SAFE:  Simple Preserved and Augmented FEatures

This is a personal reproduction of Pytorch implementation of paper:

    > [Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspectives](https://arxiv.org/abs/2408.06741)
    >
    > Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, Fuli Feng
    
    ## Requirements
    
    Install the environment as follows:
    
    ```python
    # create conda environment
    conda create -n SAFE -y python=3.9
    conda activate SAFE
    # install pytorch 
    pip install torch==2.2.1 torchvision==0.17.1
    # install other dependencies
    pip install -r requirements.txt
    ```
    
    We are using torch 2.2.1 in our production environment, but other versions should be fine as well.
    
    ## Getting the data
    
    |             |                            paper                             |                             Url                              |
    | :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
    |  Train Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [googledrive](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view) |
    |  Val   Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [googledrive](https://drive.google.com/file/d/1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL/view) |
    |  Test Set1  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [googledrive](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view) |
    |  Test Set2  | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) | [googledrive](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing) |
    |  Test Set3  | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) | [googledrive](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing) |
    |  Test Set4  | [GenImage NeurIPS2023](https://github.com/GenImage-Dataset/GenImage)             | [googledrive](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) |
    
    ## Directory structure
    
    <details>
    <summary> You should organize the above data as follows: </summary>
    
    ```
    data/datasets
    |-- train_ForenSynths
    |   |-- train
    |   |   |-- car
    |   |   |-- cat
    |   |   |-- chair
    |   |   |-- horse
    |   |-- val
    |   |   |-- car
    |   |   |-- cat
    |   |   |-- chair
    |   |   |-- horse
    |-- test1_ForenSynths/test
    |   |-- biggan
    |   |-- cyclegan
    |   |-- deepfake
    |   |-- gaugan
    |   |-- progan
    |   |-- stargan
    |   |-- stylegan
    |   |-- stylegan2
    |-- test2_Self-Synthesis/test
    |   |-- AttGAN
    |   |-- BEGAN
    |   |-- CramerGAN
    |   |-- InfoMaxGAN
    |   |-- MMDGAN
    |   |-- RelGAN
    |   |-- S3GAN
    |   |-- SNGAN
    |   |-- STGAN
    |-- test3_Ojha/test
    |   |-- dalle
    |   |-- glide_100_10
    |   |-- glide_100_27
    |   |-- glide_50_27
    |   |-- guided          # Also known as ADM.
    |   |-- ldm_100
    |   |-- ldm_200
    |   |-- ldm_200_cfg
    |-- test4_GenImage/test
    |   |-- ADM
    |   |-- BigGAN
    |   |-- Glide
    |   |-- Midjourney
    |   |-- stable_diffusion_v_1_4
    |   |-- stable_diffusion_v_1_5
    |   |-- VQDM
    |   |-- wukong
    ```
    </details>
    
    ## Training
    
    ```
    bash scripts/train.sh
    ```
    
    This script enables training with 4 GPUs, you can specify the number of GPUs by setting `GPU_NUM`.
    
    ## Inference
    
    ```
    bash scripts/eval.sh
    ```
    
    We provide the pretrained self checkpoint in `./checkpoint/checkpoint-best-self.pth`, you can directly run the script to reproduce our results. 
    
    ## Accuracy
    
    Settings: 
    
    seed=3, base_lr=1e-4, max_epochs=100, batch_size=256. SAFE-self achieves a higher overall mean accuracy (94.38% vs. 93.16%) and outperforms the original SAFE in most categories, though it shows slight trade-offs in CycleGAN, Glide, Midjourney, VQDM, and DALLE2 detection performance.
    
    
    ### AIGCDetectBenchmark
    
    |           |   ProGAN   | StyleGAN  |  BigGAN   | CycleGAN  |  StarGAN   |  GauGAN   | StyleGAN2 |   WFIR    |    ADM    |   Glide   | Midjourney |  SD v1.4  |  SD v1.5  |   VQDM    |  Wukong   |  DALLE2   |   Mean    |
    | :-------: | :--------: | :-------: | :-------: | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
    |   SAFE    |   99.86    |   98.04   |   89.72   | **98.87** |   99.90    |   91.52   |   98.57   |   51.95   |   82.05   | **96.29** | **95.27**  |   99.41   |   99.27   | **96.29** |   98.21   | **95.30** |   93.16   |
    | SAFE-self | **100.00** | **99.62** | **90.62** |   95.72   | **100.00** | **95.68** | **99.84** | **70.55** | **87.09** |   93.08   |   91.68    | **99.53** | **99.44** |   93.80   | **99.06** |   94.30   | **94.38** |
    
    &nbsp;
    
    ### ForenSynths
    
    |           |   ProGAN   | StyleGAN  | StyleGAN2 |  BigGAN   | CycleGAN  |  StarGAN   |  GauGAN   | Deepfake  |   SITD    |    SAN    |    CRN    |   IMLE    |   WFIR    |   Mean    |
    | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
    |   SAFE    |   99.86    |   98.04   |   98.57   |   89.72   | **98.87** |   99.90    |   91.52   | **93.10** | **85.56** |   95.91   | **50.10** | **50.10** |   51.95   |   84.86   |
    | SAFE-self | **100.00** | **99.62** | **99.84** | **90.62** |   95.72   | **100.00** | **95.68** |   89.82   |   81.39   | **97.95** |   50.00   |   50.00   | **70.55** | **86.25** |
    
    The numbers of images in SAN and SITD are less than 1K. ForenSynths dataset is an unbalanced dataset.
    
    &nbsp;
    
    ### Self-Synthesis (GAN Based)
    
    |           |  AttGAN   |   BEGAN   | CramerGAN | InfoMaxGAN |  MMDGAN   |  RelGAN   |   S3GAN   |   SNGAN   |   STGAN   |   Mean    |
    | :-------: | :-------: | :-------: | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
    |   SAFE    |   99.38   | **99.80** | **99.73** |   99.55    | **99.73** |   99.55   | **94.48** |   98.80   | **99.90** | **98.99** |
    | SAFE-self | **99.72** |   99.65   |   99.62   | **99.65**  |   99.65   | **99.90** |   89.35   | **99.52** |   99.85   |   98.55   |
    
    &nbsp;
    
    ### UniversalFakeDetect (DM Based)
    
    |           |   DALLE   | Glide_100_10 | Glide_100_27 | Glide_50_27 |    ADM    |  LDM_100  |  LDM_200  | LDM_200_cfg |   Mean    |
    | :-------: | :-------: | :----------: | :----------: | :---------: | :-------: | :-------: | :-------: | :---------: | :-------: |
    |   SAFE    |   97.50   |  **97.25**   |  **95.75**   |  **96.60**  |   82.36   |   98.80   |   98.80   |    98.65    | **95.71** |
    | SAFE-self | **98.30** |    94.85     |    92.10     |    95.40    | **82.95** | **99.55** | **99.60** |  **99.55**  |   95.29   |
    
    &nbsp;
    
    ### GenImage
    
    |           | Midjourney |  SDv1.4   |  SDv1.5   |    ADM    |   Glide   |  Wukong   |   VQDM    |  BigGAN   |   Mean    |
    | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
    |   SAFE    | **95.27**  |   99.41   |   99.27   |   82.05   | **96.29** |   98.21   | **96.29** | **97.84** | **95.58** |
    | SAFE-self |   91.68    | **99.53** | **99.44** | **87.09** |   93.08   | **99.06** |   93.80   |   96.33   |   93.15   |
    
    &nbsp;
    
    
    
    ## Citing
    
    If you find this repository useful for your work, please consider citing it as follows:
    ```
    @article{li2024improving,
      title={Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective},
      author={Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Feng, Fuli},
      journal={arXiv preprint arXiv:2408.06741},
      year={2024}
    }
    ```
    
