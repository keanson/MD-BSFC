python calibrate.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --steps=3 --tau=1 --loss="ELBO" --seed=53
python calibrate.py --dataset="cross" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --steps=3 --tau=1 --loss="ELBO" --seed=1
python calibrate.py --dataset="cross_char" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --steps=3 --tau=1 --loss="ELBO" --seed=727 --kernel='rbf'