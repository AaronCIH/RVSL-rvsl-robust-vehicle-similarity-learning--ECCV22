# RVSL--ECCV2022
This is official implentation of the paper:
"RVSL: Robust Vehicle Similarity Learning in Real Hazy Scenes Based on Semi-supervised Learning"

## 1. Abstract:
Recently, vehicle similarity learning, also called re-identification (ReID), has attracted significant attention in computer vision. Several algorithms have been developed and obtained considerable success. However, most existing methods have unpleasant performance in the hazy scenario due to poor visibility. Though some strategies are possible to resolve this problem, they still have room to be improved due to the limited performance in real-world scenarios and the lack of real-world clear ground truth. Thus, to resolve this problem, inspired by CycleGAN, we construct a training paradigm called RVSL which integrates ReID and domain transformation techniques. The network is trained on semi-supervised fashion and does not require to employ the ID labels and the corresponding clear ground truths to learn hazy vehicle ReID mission in the real-world haze scenes. To further constrain the unsupervised learning process effectively, several losses are developed. Experimental results on synthetic and real-world datasets indicate that the proposed method can achieve state-of-the-art performance on hazy vehicle ReID problems. It is worth mentioning that although the proposed method is trained without real-world label information, it can achieve competitive performance compared to existing supervised methods trained on complete label information.

You can also refer our previous works on other defoggy vehicle reid applications! <br />
**SJDL-Vehicle: Semi-supervised Joint Defogging Learning for Foggy Vehicle Re-identification--AAAI2022.** [[Link]](https://github.com/Cihsaing/SJDL-Foggy-Vehicle-Re-Identification--AAAI2022)

## 2. Network Architecture
* Inspired by the cycle consistency, proposed RVSL framework:
Learning knowledge of transformation of two scenarios through synthetic data, and then enhance the model’s robustness in real-world scenes in an unsupervised manner.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/cycle_consistency.png)

* Robust Vehicle Similarity Learning
RVSL based on semi-supervised learning and domain transformation is proposed to learn hazy vehicle ReID without the albels or clear ground truths of real-world data.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/architecture.png)

## 3. Dataset 
Both synthetic data and real-world data are adopted in this paper, and this data is constructed by myself, the detail can see [Dataset](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/tree/master/Datasets):
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/dataset.png)

## 4. Result
* For synthetic hazy dataset.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/Syn_SOTA.png)

* For real-world hazy dataset.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/Real_SOTA.png)

# Implement
## 1. Setup and environment
To implement our method you need:
> 1. Python 3.10
> 2. pytorch 1.8.0+
> 3. torchvision 0.13.0+
> 4. yacs
> 5. tqdm

* Devices
The network is trained on an Nvidia Tesla V100 Multi-GPUs for 3 days, and the proposed code is based on single gpu which needs bigger gpu memory.
**If you encounter the problem "out of the memory", please transfer the training code to multi-gpus or low the config "DATALOADER.NUM_INSTANCE" and "SOLVER.IMS_PER_BATCH"**

## 2. Data Preparation
Since the policy of Veriwild and Vehicle1M, we can only provide the codes to synthesize the foggy data and the index of the real-world foggy data. Please follow the steps to generate the data:
See [Data Preparation](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/tree/master/Datasets).

## 3. Train RVSL
Run following command to train the complete RVSL model
```
cd RVSL/
bash train.sh
```

You can also train the model seperately.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python <TRAINER> -c configs/FVRID.yml MODE.STAGE <STAGE> MODEL.PRETRAIN_PATH <PRETRAIN> OUTPUT_DIR <OUTPUT_PATH>
```
where the ```<gpu_id>``` is assigned gpu number. <br>
where the ```<TRAINER>``` is the trainer file. {stage1_trainer.py, stage2_trainer.py, stage3_trainer.py} <br>
where the ```<STAGE>``` is the training stage. <br>
-> ("STAGE1":supervised training stage, "STAGE2":unsupervised real clear training stage, "STAGE3":unsupervised real hazy training stage) <br>
where the ```<PRETRAIN>``` is the pretrained weights path. <br>
where the ```<OUTPUT_PATH>``` is the output paths. <br>

### Common problem
1. "cuda: out of memory": the original model is trained on multi-gpu, please rewrite the trainer code to suit for parallel training.
2. "cuda: out of memory": if you only have a gpu, please reduce the config "DATALOADER.NUM_INSTANCE" and "SOLVER.IMS_PER_BATCH", e.g. "SOLVER.IMS_PER_BATCH" must to be multiply of NUM_INSTANCE.
3. If you encounter an amp conflict, there are two possibilities: torch version problem and the device must have support.
   If your device not support, please keep the ```"configs/FVRID_syn.yml": SOLVER.FP16 = False```.

## Pretrained Models
We provide the pretrained SJDL, training on FVRID for your convinient. You can download it from the following link: 
[https://drive.google.com/file/d/1WhsvYQP-qg1R-BcpH5lonjxh4DYp2ouv/view?usp=sharing](https://drive.google.com/drive/folders/1d8Ggtc5GHA7L5Mrv8-teyTPA4TO3A8MA?usp=sharing)

## Testing
Run following command to test the complete result.
```
cd RVSL/
bash test.sh
```

You can also train the model seperately.
```
CUDA_VISIBLE_DEVICES=<gpu_id> python inference.py -t -c configs/FVRID.yml TEST.DATA <Data> MODE.STAGE "STAGE1" TEST.WEIGHT <PTH_PATH> OUTPUT_DIR <OUTPUT_PATH> 
```
where the ```<gpu_id>``` is assigned gpu number. <br>
where the ```<Data>``` is used to select testing set. {"SYN", "REAL_CLEAR", "REAL_FOGGY"} <br>
where the ```<PTH_PATH>``` is the test weight. <br>
where the ```<OUTPUT_PATH>``` is the output paths. <br>

The pre-trained model can be downloaded from Link: <br>
https://drive.google.com/file/d/1WhsvYQP-qg1R-BcpH5lonjxh4DYp2ouv/view?usp=sharing. <br>
and you can put it at the dir ```'./RVSL/output/'```

# Citations
Please cite this paper in your publications if it is helpful for your tasks:    

Bibtex:
```
@inproceedings{,
  title={},
  author={},
  booktitle={},
}
```
