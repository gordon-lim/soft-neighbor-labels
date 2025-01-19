# **Label Memorization of Human Noisy Labels** 🧠  
This folder provides everything you need to calculate **label memorization scores** for the CIFAR-10N human noisy labels, as described in our paper. The analysis and figure generation are covered in the `analysis` Jupyter notebook.

---

## **Step 1: Get CIFAR-10N files** 📈  

a. Clone the CIFAR-10N repo
   ```bash
   git clone https://github.com/UCSC-REAL/cifar-10-100n.git
   ```
b. Move our files over
   ```bash
   mv cifar_infl_mem.py aggregate_infl_mem.py cifar-10-100n/
   ```
## **Step 2: Calculate Label Memorization**   
This script (`cifar_infl_mem.py`) computes **influence memorization** scores for the CIFAR-10 dataset by training models on held-out subsets of noisy and clean labels. The method is adapted from Google's Held-Out Influence Estimation framework but has been converted to PyTorch and tailored for CIFAR-10/100 datasets.  

---

### **Arguments** 🛠️  

   Below are the configurable arguments for the script:
   
   | Argument          | Type      | Default       | Description                                                                                  |
   |--------------------|-----------|---------------|----------------------------------------------------------------------------------------------|
   | `noise_type`     | `str`     | `clean`       | Type of noise to use. Options: `clean`, `aggre`, `worst`, `rand1`, `rand2`, `rand3`, `clean100`, `noisy100`. |
   | `noise_path`     | `str`     | `None`        | Path to the CIFAR-10 human noise labels file (e.g., `CIFAR-10_human.pt`).                    |
   | `dataset`        | `str`     | `cifar10`     | Dataset to use. Options: `cifar10` or `cifar100`.                                            |
   | `is_human`       | `flag`    | `False`       | If specified, uses human noise labels for the experiment.                                     |
   | `seed_start`     | `int`     | **Required**  | Starting seed for training models.                                                           |
   | `seed_end`       | `int`     | **Required**  | Ending seed for training models.                                                             |
   | `gpu`            | `int`     | `0`           | GPU index to use for computation.                                                           |
   
   ---

   `--seed_start` and `--seed_end` allow you to define a specific range of seeds for model training, making it easier to checkpoint and manage large-scale experiments. In our case, we incrementally trained 100 models at a time before aggregating their results, as shown in Step 3.
   
   ### **Example Command** 📋  
   
   To compute influence memorization scores for the **human** `rand1` noise type using seeds 0 through 500 incrementally (100 at a time) on GPU 0:
   
   ```bash
   python cifar_infl_mem.py --noise_type rand1 --is_human True \
   --dataset cifar10 --seed_start 0 --seed_end 99 --gpu 0

   python cifar_infl_mem.py --noise_type rand1 --is_human True \
   --dataset cifar10 --seed_start 100 --seed_end 199 --gpu 0
   ```

   ### **Directory Tree**  📂

   After running the script, the results will be saved in the following directory structure under `results_infl_mem/`:
   ```
      results_infl_mem/
   ├── cifar-rand1-human-0-99-infl-mem.npz
   └── cifar-rand1-human-100-199-infl-mem.npz
   ```

   Each file corresponds to the results from a specific range of seeds.

## **Step 3: Aggregate results**

   Once all desired models have been trained, aggregate the results using aggregate_infl_mem.py:

   ```bash
   python aggregate_infl_mem.py --results_dir results_infl_mem --is_human True
   ```

   ### Output Example: ##
   ```bash
   results_infl_mem/cifar-rand1-human-infl-mem.npz 
   ```
   This file contains the combined influence memorization scores for all runs (e.g., 200 total seeds in the example above).

   **FIXME:** Since we only worked with `rand1` human/synthetic noise, the aggregate_infl_mem.py script is hardcoded to only differentiate between those result files. The aggregated file will always be saved with the format: `cifar-rand1-{human/syn}-infl-mem.npz`. 

   ___

   Once you have the aggregated results, you can run our analysis code for visualization, as described in our paper.

   Happy analyzing! 🎉





   
