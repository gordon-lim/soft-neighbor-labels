for seed in 1 2 3 
do
    python train_CIFAR.py --noise_type cifar10-cluster-0.35 --seed ${seed} --device cuda:0
done


