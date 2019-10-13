python train.py LR 6 --seed 9487
python train.py LR 11 --seed 9487
python train.py LR 18 --seed 9487
python train.py NN 6 --seed 9487 --dm 16 --dropout 0.1 --depth 10 --residual_type DenseNet_1 --act_fn relu
python train.py NN 11 --seed 9487 --dm 16 --dropout 0.1 --depth 10 --residual_type DenseNet_1 --act_fn relu
python train.py NN 18 --seed 9487 --dm 16 --dropout 0.1 --depth 10 --residual_type DenseNet_1 --act_fn relu
