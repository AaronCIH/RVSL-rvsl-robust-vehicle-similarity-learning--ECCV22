# Testing RVSL models on two datasets
# Dataset: FVRID_syn, FVRID_real (real foggy)
# imagesize: 384x384
# inference: TEST.DATA {"SYN", "REAL_CLEAR", "REAL_FOGGY"}
# * Please download pretrained models, and put it into ./output/ folder.
# * use STAGE1 model, dual path forward.
CUDA_VISIBLE_DEVICES=1 python inference.py -t -c configs/FVRID.yml TEST.DATA "SYN" MODE.STAGE "STAGE1" TEST.WEIGHT "./output/RVSL.pth" OUTPUT_DIR "./output/RVSL/TEST_SYN" 
CUDA_VISIBLE_DEVICES=1 python inference.py -t -c configs/FVRID.yml TEST.DATA "REAL_FOGGY" MODE.STAGE "STAGE1" TEST.WEIGHT "./output/RVSL.pth" OUTPUT_DIR "./output/RVSL/TEST_REAL_FOGGY" 

CUDA_VISIBLE_DEVICES=1 python inference.py -t -c configs/FVRID.yml TEST.DATA "SYN" MODE.STAGE "STAGE1" TEST.WEIGHT "./output/RVSL_Sup.pth" OUTPUT_DIR "./output/RVSL-sup/TEST_SYN" 
CUDA_VISIBLE_DEVICES=1 python inference.py -t -c configs/FVRID.yml TEST.DATA "REAL_FOGGY" MODE.STAGE "STAGE1" TEST.WEIGHT "./output/RVSL_Sup.pth" OUTPUT_DIR "./output/RVSL-sup/TEST_REAL_FOGGY" 