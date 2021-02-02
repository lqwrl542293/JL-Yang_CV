# BiSeNet v2
# USSD
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start=0.000301,
    weight_decay=5e-4,
    warmup_iters=1000,
    max_iter=60000,

    # 数据集类别数 (used by train_amp.py, evaluate.py, export_onnx.py)
    n_classes=6,
    # 数据集的mean、std (BGR的顺序)
    mean_value=(0.5391, 0.6011, 0.6022),
    std_value=(0.1888, 0.1599, 0.1751),

    im_root=r'/root/Datasets/USSD',
    train_im_anns=r'/root/BiSeNet_v1_v2_PyTorch/datasets/USSD/train.txt',
    val_im_anns=r'/root/BiSeNet_v1_v2_PyTorch/datasets/USSD/test_resized_2.txt',
    save_model_freq=1000,
    scales=[0.75, 2.],
    #     cropsize=[540, 960],  540不行，因为不是32的整数倍
    cropsize=[544, 960],
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
    respth='/root/BiSeNet_v1_v2_PyTorch/zyh/labs/ussd_v2_0113_finetune',

    # evaluate时进行crop的crop size (used by evaluate.py)
    # eval_crop_size=[1080, 1920],
    eval_crop_size=[544, 960],
    # onnx模型接受的输入的尺寸, [height, width] (used by export_onnx.py):
    onnx_input_size=[544, 960],
    # trt engine接受的输入的尺寸, [height, width] (used by export_onnx.py):
    trt_engine_input_size=[544, 960]
)
