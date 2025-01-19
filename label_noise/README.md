
# **Cluster-Based Noise** 🎯  
This folder provides details about how to generate **cluster-based noise** for your own datasets and describes the noise label files we've already generated for the CIFAR-10 and CIFAR-100 datasets.  

---

## **1. Generating Cluster-Based Noise** 🛠️  

To generate cluster-based noise, use the script `cluster_based_noise.py`. Below are the required inputs and parameters:  

### **Arguments for `cluster_based_noise.py`**  
| Argument                  | Description                                                                                 |
|---------------------------|---------------------------------------------------------------------------------------------|
| `dataset`   | ame of the dataset (e.g., cifar10 or cifar100) for output filename.                       |
| `embeddings_file`   | Path to the `.pkl` file containing CLIP embeddings for the dataset.                         |
| `labels_file`            | Path to the `.npy` file with the dataset's labels.                                          |
| `n`                     | Number of clusters to generate noise from (higher values = more noise).                     |
| `r`                     | Radius multiplier to determine how far noisy labels can spread from their cluster centers. |

We’ve provided:  
- The script to generate CLIP embeddings: `clip_embeddings.py`.  
- Precomputed CLIP embeddings for CIFAR-10 and CIFAR-100 as used in our paper.  

**Command Example:**  
To generate cluster-based noise with `n=57` and `r=2` for CIFAR-10:  
```bash
python cluster_based_noise.py --dataset cifar10 --embeddings_file cifar10-clip-embeddings.pkl \
--labels_file cifar10_labels.npy --n 57 --r 2
```

---

## **2. Noise Label Files** 📂  

We’ve already generated several noise label files for CIFAR-10 and CIFAR-100 based on different parameters. Below are the configurations used:

### **Cluster-Based Noise (CIFAR-10)**  
- **Noise Level: 0.35**  
  Parameters: `n=57`, `r=2`  
  File: `cifar10_cluster_noise_35.npy`  
- **Noise Level: 0.70**  
  Parameters: `n=190`, `r=2`  
  File: `cifar10_cluster_noise_70.npy`  

### **Cluster-Based Noise (CIFAR-100)**  
- **Noise Level: 0.35**  
  Parameters: `n=23`, `r=1.2`  
  File: `cifar100_cluster_noise_35.npy`  
- **Noise Level: 0.70**  
  Parameters: `n=80`, `r=1.5`  
  File: `cifar100_cluster_noise_70.npy`  

### **PMD Noise** 🔄  
For **PMD noise**, we used the configurations from the [original repository](https://github.com/pxiangwu/PLC) (all are type 1).  

---

## **File Formats** 📜  

Some methods in `lnl_methods/` use `.csv` files, while others use `.npy`. To accommodate both, we provide:  
- Noise label files in **both `.csv` and `.npy` formats**.  
- Conversion scripts for converting between formats.  

---

With these resources, you can generate and use cluster-based noise for robust model training, as demonstrated in our paper. **Happy testing!** 🎉  

