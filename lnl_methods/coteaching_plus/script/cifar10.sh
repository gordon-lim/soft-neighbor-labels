for seed in 1 2 3 
do
  for model_type in coteaching_plus 
  do
    python main.py --dataset cifar10 --model_type ${model_type} --noisy_label_file ../label_noise/cifar10-cluster-0.70.csv --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset cifar10 --model_type ${model_type} --noisy_label_file ../label_noise/cifar10-pmd-0.70.csv --seed ${seed} --result_dir results/trial_${seed}/
  done
done
