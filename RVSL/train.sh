# Training RVSL models with three stages
# Dataset: FVRID
# imagesize: 384x384
# batchsize: 6x6
# inference: TEST.DATA {"SYN", "REAL_CLEAR", "REAL_FOGGY"}
CUDA_VISIBLE_DEVICES=0 python stage1_trainer.py -c configs/FVRID.yml MODE.STAGE "STAGE1" MODEL.PRETRAIN_PATH "" OUTPUT_DIR "./output/RVSL_Stage1/"
CUDA_VISIBLE_DEVICES=0 python stage2_trainer.py -c configs/FVRID.yml MODE.STAGE "STAGE2" MODEL.PRETRAIN_PATH "./output/RVSL_Stage1/checkpoint/best.pth" OUTPUT_DIR "./output/RVSL_Stage2/"
CUDA_VISIBLE_DEVICES=0 python stage3_trainer.py -c configs/FVRID.yml MODE.STAGE "STAGE3" MODEL.PRETRAIN_PATH "./output/RVSL_Stage2/checkpoint/best.pth" OUTPUT_DIR "./output/RVSL_Stage3/"
CUDA_VISIBLE_DEVICES=0 python inference.py -t -c configs/FVRID.yml TEST.DATA "REAL_FOGGY" MODE.STAGE "STAGE1" TEST.WEIGHT "./output/RVSL_Stage3/checkpoint/best.pth" OUTPUT_DIR "./output/RVSL_Stage3/TEST_REAL_FOGGY" TEST.VIS True