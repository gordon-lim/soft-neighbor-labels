for seed in 1 2 3 
do
    python train.py --noisy_label_file ../../label_noise/cifar10-cluster-0.70.npy --seed ${seed}
    python train.py --noisy_label_file ../../label_noise/cifar10-pmd-0.70.npy --seed ${seed}
done
