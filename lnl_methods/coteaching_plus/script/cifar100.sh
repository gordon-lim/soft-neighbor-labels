for seed in 1 2 3 
do
  for model_type in coteaching_plus 
  do
    python main.py --dataset cifar100 --model_type ${model_type} --noisy_label_file ../label_noise/cifar100-cluster-0.70.csv --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset cifar100 --model_type ${model_type} --noisy_label_file ../label_noise/cifar100-pmd-0.70.csv --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset cifar100 --model_type ${model_type} --noisy_label_file ../label_noise/cifar100-cluster-0.35.csv --seed ${seed} --result_dir results/trial_${seed}/
    python main.py --dataset cifar100 --model_type ${model_type} --noisy_label_file ../label_noise/cifar100-pmd-0.35.csv --seed ${seed} --result_dir results/trial_${seed}/
  done
done