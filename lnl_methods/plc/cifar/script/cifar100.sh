for seed in 1 2 3 
do
    python train.py --noisy_label_file ../../label_noise/cifar100-cluster-0.35.npy --seed ${seed} --gpus 1
    python train.py --noisy_label_file ../../label_noise/cifar100-pmd-0.35.npy --seed ${seed} --gpus 1
    python train.py --noisy_label_file ../../label_noise/cifar100-cluster-0.70.npy --seed ${seed} --gpus 1
    python train.py --noisy_label_file ../../label_noise/cifar100-pmd-0.70.npy --seed ${seed} --gpus 1
done
