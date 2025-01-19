for seed in 1 2 3 
do
    python trainer.py --dataset cifar100 --noisy_label_file cifar100-cluster-0.35.csv --loss_function gce --seed ${seed} --gpu 1
    python trainer.py --dataset cifar100 --noisy_label_file cifar100-pmd-0.35.csv --loss_function gce --seed ${seed} --gpu 1
    python trainer.py --dataset cifar100 --noisy_label_file cifar100-cluster-0.70.csv --loss_function gce --seed ${seed} --gpu 1
    python trainer.py --dataset cifar100 --noisy_label_file cifar100-pmd-0.70.csv --loss_function gce --seed ${seed} --gpu 1
done