
# BiSeNet v2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start=5e-2,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=432000,
    
    n_classes=6,
    # 数据集的mean、std (BGR的顺序)
    mean_value=(0.3331, 0.3597, 0.3365),
    std_value=(0.1432, 0.1378, 0.1405),
    
    # TODO: 数据集的根目录及索引文件的路s径
    im_root=r'/root/Datasets/Potsdam',
    train_im_anns=r'/root/BiSeNet_v1_v2_PyTorch/datasets/Potsdam/train.txt',
    val_im_anns=r'/root/BiSeNet_v1_v2_PyTorch/datasets/Potsdam/test.txt',
    save_model_freq=2000,
    
    scales=[0.25, 2.],
    cropsize=[512, 512],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,

    # 模型文件和evaluate结果保存路径:
    respth='/root/BiSeNet_v1_v2_PyTorch/zyh/labs/v2_0106',
)
