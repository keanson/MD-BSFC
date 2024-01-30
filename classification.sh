python train.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python train.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python train.py --dataset="cross_char" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO" --kernel="rbf"
python train.py --dataset="cross_char" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO" --kernel="rbf"
python train.py --dataset="cross" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python train.py --dataset="cross" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python test.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python test.py --dataset="CUB" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python test.py --dataset="cross_char" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO" --kernel="rbf"
python test.py --dataset="cross_char" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO" --kernel="rbf"
python test.py --dataset="cross" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=1 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"
python test.py --dataset="cross" --method="MD" --train_n_way=5 --test_n_way=5 --n_shot=5 --seed=1 --train_aug --steps=3 --tau=1 --loss="ELBO"