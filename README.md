# RVSL--ECCV2022
This is official implentation of the paper:
RVSL: Robust Vehicle Similarity Learning in Real Hazy Scenes Based on Semi-supervised Learning -- ECCV2022

## Abstract:
Recently, vehicle similarity learning, also called re-identification (ReID), has attracted significant attention in computer vision. Several algorithms have been developed and obtained considerable success. However, most existing methods have unpleasant performance in the hazy scenario due to poor visibility. Though some strategies are possible to resolve this problem, they still have room to be improved due to the limited performance in real-world scenarios and the lack of real-world clear ground truth. Thus, to resolve this problem, inspired by CycleGAN, we construct a training paradigm called RVSL which integrates ReID and domain transformation techniques. The network is trained on semi-supervised fashion and does not require to employ the ID labels and the corresponding clear ground truths to learn hazy vehicle ReID mission in the real-world haze scenes. To further constrain the unsupervised learning process effectively, several losses are developed. Experimental results on synthetic and real-world datasets indicate that the proposed method can achieve state-of-the-art performance on hazy vehicle ReID problems. It is worth mentioning that although the proposed method is trained without real-world label information, it can achieve competitive performance compared to existing supervised methods trained on complete label information.

You can also refer our previous works on other defoggy vehicle reid applications!
SJDL-Vehicle: Semi-supervised Joint Defogging Learning for Foggy Vehicle Re-identification--AAAI2022. [[Link]](https://github.com/Cihsaing/SJDL-Foggy-Vehicle-Re-Identification--AAAI2022)

## Network Architecture
* Inspired by the cycle consistency, proposed RVSL framework:
Learning knowledge of transformation of two scenarios through synthetic data, and then enhance the model’s robustness in real-world scenes in an unsupervised manner.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/cycle_consistency.png)

* Robust Vehicle Similarity Learning
RVSL based on semi-supervised learning and domain transformation is proposed to learn hazy vehicle ReID without the albels or clear ground truths of real-world data.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/architecture.png)

## Dataset 
Both synthetic data and real-world data are adopted in this paper, and this data is constructed by myself, the detail can see [Dataset](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/tree/master/Datasets):
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/dataset.png)

## Result
* For synthetic hazy dataset.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/Syn_SOTA.png)

* For real-world hazy dataset.
![image](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/raw/master/Fig/Real_SOTA.png)


# Setup and environment
To implement our method you need:
> 1. Python 3.10
> 2. pytorch 1.8.0+
> 3. torchvision 0.13.0+
> 4. yacs
> 5. tqdm

* Devices
The network is trained on an Nvidia Tesla V100 Multi-GPUs for 3 days, and the proposed code is based on single gpu which needs bigger gpu memory.
**If you encounter the problem "out of the memory", please transfer the training code to multi-gpus or low the config "DATALOADER.NUM_INSTANCE" and "SOLVER.IMS_PER_BATCH"**

## Data Preparation
Since the policy of Veri-1M, we can only provide the codes to synthesize the foggy data and the index of the real-world foggy data. Please follow the steps to generate the data:
See [Data Preparation](https://github.com/Cihsaing/rvsl-robust-vehicle-similarity-learning--ECCV22/tree/master/Datasets).

## Train RVSL
Run following command to train the RVSL model
```
cd RVSL/

```

### Common problem


## Pretrained Models

## Testing
```

```
where the ```<Configs>``` is the testing configs file. <br>
where the ```<PTH_PATH>``` is the test weight. <br>
where the ```<OUTPUT_PATH>``` is the output paths. <br>

The pre-trained model can be downloaded from Link: <br>
. <br>
and you can put it at the dir ```''```

Examples
```

```

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
