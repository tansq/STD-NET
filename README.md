# STD-NET: Search of Image Steganalytic Deep-learning Architecture via Hierarchical Tensor Decomposition

Implementation of our paper

Directories and files included in the implementation:


'datas/' - Some of the datasets of our experiment.

'libs/' - Functional code.

'prep/' - Code to generate .mat files.

'tasks/' - Task-related scripts. 


'third_party' - Codes implemented by J. Fridrich.

'generated_cfg_and_model/' - Generated config files and corresponding '.ckpt' models.

* The generated config files and the corresponding well trained models are listed below:

| cfg                              | ckpt                                                         |
| -------------------------------- | ------------------------------------------------------------ |
| cfgs/bb_qf75ju04_cylinder.cfg    | models/std_bb_qf75ju04/bb_qf75ju04_cylinder/;<br>models/alav2_qf75ju04_base_std_bb_qf75ju04/cylinder/ |
| cfgs/bb_qf75ju04_ladder.cfg      | models/std_bb_qf75ju04/bb_qf75ju04_ladder;<br>models/alav2_qf75ju04_base_std_bb_qf75ju04/ladder/ |
| cfgs/bb_qf90JU04_cylinder.cfg    | models/std_bb_qf90ju04/scratch_cylinder/                     |
| cfgs/bb_qf90JU04_ladder.cfg      | models/std_bb_qf90ju04/scratch_ladder/                       |
| cfgs/bb512_qf75ju04_cylinder.cfg | models/std_bb512_qf75ju04/scratch-cylinder/                  |
| cfgs/bb512_qf75ju04_ladder.cfg   | models/std_bb512_qf75ju04/scratch-ladder/                    |



## Requirement
* python 3.6+
* tensorflow 1.12.0
* tensorly 0.4.5


## Demo

### tensor decomposition 
1. train a SRNetC64 model to be decomposed
```bash
python 1_train_val.py --log_root '../../log_dir/' --work_name '1_train_val_srnetc64' --data_path './datasets/'
```

2. traverse trained SRNetC64 model to determine normalized distortion

```bash
python 3_inspect_ma.py --log_root '../../log_dir/' --work_name '3_inspect_ma' --load_path 'trained SRNetC64 model path' --data_path './datasets/'
```

3. detect to search cylinder/ladder-shaped STD-NET models
```bash
python 4_detect.py --log_root '../../log_dir/' --work_name '4_detect' --data_path './datasets/' --load_path 'trained SRNetC64 model path'
```

4. decompose the SRNetC64 model and save the decomposed model that preserve parameters.

```bash
python 5_decom.py --log_root '../../log_dir/' --work_name '5_decom' --data_path './datasets/' --load_path 'trained SRNetC64 model path'
```

4. train STD-NET model from scratch or finetune the STD-NET model that preserve the model parameters after Tucker decomposition.
```bash
python 7_train_from_scratch_decom.py --log_root '../../log_dir/' --work_name '7_train_from_scratch' --data_path './datasets/' --load_config '../../generated_cfg_and_model/cfgs/bb_qf75ju04_cylinder.cfg'
```

```bash
python 6_finetune_decom_srnet.py --log_root '../../log_dir/' --work_name '6_finetune_decom_srnet' --load_path 'Path of the decompose model that preserve parameters' --data_path './datasets/' --load_config '../../generated_cfg_and_model/cfgs/bb_qf75ju04_cylinder.cfg'
```


### Evaluate
1. test the decomposed model trained from scratch
```bash
python 8_test_decom.py --log_root '../../log_dir/' --work_name '8_test_decom' --data_path './datasets/' --load_path '../../generated_cfg_and_model/models/bb_qf75ju04_cylinder/train_from_scratch_tucker_429625.ckpt' --load_config '../../generated_cfg_and_model/cfgs/bb_qf75ju04_cylinder.cfg'
```



