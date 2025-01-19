for seed in 1 2 3 
do
    python trainer.py --noisy_label_file cifar10-cluster-0.70.csv --loss_function gce --seed ${seed} --gpu 1
    python trainer.py --noisy_label_file cifar10-pmd-0.70.csv --loss_function gce --seed ${seed} --gpu 1
done