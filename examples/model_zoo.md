## Training and testing shell scripts for dynamic PCC models
### Part 1: Dataset Catalog Structure
#### 1.Catalog Structure:  
    └── zyDatasets/
        ├── dynamic_data/           # Dataset used for training (dominated by 8iVFB)
        │   ├── vox_10_kdtree/      # Sequence after kd-tree partitioning
        │   │   ├── longdress/
        │   │   ├── loot/
        │   │   ├── queen/
        │   │   ├── redandblack/
        │   │   ├── soldier/
        |   |   └── ...
        │   └── vox_10/             # Original sequence
        │       ├── longdress/
        │       ├── loot/
        │       ├── queen/
        │       ├── redandblack/
        │       ├── soldier/
        │       └── ...
        └── MPEG_GPCC_CTC/
            └── Dynamic/
                ├── basketball_player/
                ├── dancer/
                ├── exercise/
                ├── model/
                └── ...

#### 2.Sequence Naming Rules:  
```
{sequence name}_vox{resolution}_{frame number}.ply  

For example:  
longdress_vox10_1051.ply   
longdress_vox10_1052.ply   
......
```

#### 3.Notes
In order to train and test the dataset in a uniform form, we need to make a small modification to the way D-DPCC and LDPCC load the point cloud data,please refer to the ```patchs/```:



### Part 2: Training shell scripts
#### 1.D-DPCC
```shell
cd D-DPCC
```
Train D-DPCC models
```shell
python trainer.py --batch_size=4 --gpu=0 --lamb=10 --exp_name=I10 --epoch=10 --dataset_dir='/data/zyDatasets/dynamic_data/vox_10/'
```
Train lossless model for the compression of 2x downsampled coordinates (already given)
```shell
python trainer_lossless.py --dataset_dir='/data/zyDatasets/dynamic_data/vox_10/'
```

#### 2.LDPCC
```shell
cd LDPCC
```

```shell
python -u trainer.py --pretrained=./pretrain_ckpts/I15_best_model.pth --lamb=10 --exp_name=I10 --gpu=3 >3.out 2>&1 &
```
#### 3.dynamic4d
```shell
cd dynamic4d
```
Executed in a terminal: command form I:
```shell
 python trainFrames2.py config/convolutional/coord_dynamic4d/baseline_8Stage_r2_2frames.yaml train.device='0' train.rundir_name='baseline4dimLastRefine/baseline_8stage_inter_lossy_kdTreeScaling_crossfeaEmbedding_data8iKdtree_epochs20_r2' train.batch_size=4 train.grad_acc_steps=2
```

Executed in a terminal: command form II:
```shell
./runs.sh
```

#### 4.PCGCv2

```shell
cd PCGCv2
```

```shell
python train.py --dataset='training_dataset_rootdir'   
```

#### 5.FastPCC

```shell
cd FastPCC
```
Basic testing orders:
```shell
python test.py config/convolutional/lossy_coord_v2/baseline_r1.yaml test.from_ckpt='weights/convolutional/lossy_coord_v2/baseline_r1.pt' test.device='0'
```

#### 6.SparsePCGC
```shell
cd SparsePCGC
```

See ```train/README_train.md``` for detailed commands, such as:
```shell
python train.py --dataset='../../dataset/shared/wjq/dataset/ShapeNet/pc_vox8_n100k/'  --dataset_test='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' --stage=8 --channels=32 --kernel_size=3 --scale=4 --enc_type='pooling' --epoch=30 --batch_size=4 --augment --init_ckpt='../ckpts/dense/epoch_last.pth' --only_test 
```

### Part 3: Testing shell scripts
#### 1.D-DPCC
```shell
cd D-DPCC
```

Estimate the bitrate with factorized entropy model, without practical and separate encoding and decoding process:

```shell
python test_owlii.py --log_name='results/log' --gpu=0 --frame_count=32 --results_dir='results' --tmp_dir='tmp' --dataset_dir='/data/zyDatasets/MPEG_GPCC_CTC/Dynamic'
```

With separate encoding and decoding process, which generates real bitstream, and calculate encoding and decoding time.

```shell
python test_time.py --log_name='results/log' --gpu=0 --frame_count=32 --results_dir='results' --tmp_dir='tmp' --dataset_dir='/data/zyDatasets/MPEG_GPCC_CTC/Dynamic'
```

#### 2.LDPCC
```shell
cd LDPCC  
chmod -R 777 ./
```

```shell
python -u new_test_owlii_mpeg.py --log_name=mpeg-results-96-3/96frames-3 --tmp_dir=tmp-3 --gpu=0 --results_dir=mpeg-results-96-3 --dataset_dir='/data/zyDatasets/MPEG_GPCC_CTC/Dynamic' --frame_count=32 --overwrite=True
```

#### 3.dynamic4d
```shell
cd dynamic4d
```
Executed in a terminal: command form I:
```shell
python3 test.py ./config/convolutional/coord_dynamic4d/baseline_8Stage_r2_2frames.yaml test.weights_from_ckpt='/home/zy/project/dynamic4d/4dCheckpoint/r2/ckpts/epoch_19.pt'
```

Executed in a terminal: command form II:
```shell
./test.sh
```

#### 4.PCGCv2

```shell
cd PCGCv2
```

```shell
sudo chmod 777 tmc3 pc_error_d
python coder.py --filedir='longdress_vox10_1300.ply' --ckptdir='ckpts/r3_0.10bpp.pth' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='longdress_vox10_1300.ply' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='dancer_vox11_00000001.ply'--scaling_factor=1.0 --rho=1.0 --res=2048
python test.py --filedir='Staue_Klimt_vox12.ply' --scaling_factor=0.375 --rho=4.0 --res=4096
python test.py --filedir='House_without_roof_00057_vox12.ply' --scaling_factor=0.375 --rho=1.0 --res=4096
```

#### 5.FastPCC

```shell
cd FastPCC
```
Basic training orders:
```shell
python train.py config/convolutional/lossy_coord_v2/baseline_r1.yaml train.device='0' train.rundir_name='lossy_coord_v2/baseline_r1'
```

#### 6.SparsePCGC

```shell
cd SparsePCGC
```

See ```README.md``` for detailed commands, such as (for dense point cloud):
```bash
# dense lossless
python test_ours_dense.py --mode='lossless' \
--ckptdir='../ckpts/dense/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' \
--prefix='ours_8i'
```

```bash
# dense lossy
python test_ours_dense.py --mode='lossy' \
--ckptdir='../ckpts/dense/epoch_last.pth' \
--ckptdir_sr='../ckpts/dense_1stage/epoch_last.pth' \
--ckptdir_ae='../ckpts/dense_slne/epoch_last.pth' \
--filedir='../../dataset/testdata/testdata_sparsepcgc/8iVFB/' \
--psnr_resolution=1023 --prefix='ours_8i_lossy'
```
