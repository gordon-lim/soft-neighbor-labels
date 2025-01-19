import os
import argparse
import numpy as np

def aggregate_results(results_dir, is_human):
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.npz')]
    
    aggregated_results = []
    
    human_flag = "human" if is_human else "syn"
    
    for file in result_files:
        if human_flag in file:
            data = np.load(os.path.join(results_dir, file))
            aggregated_results.append(data)
    
    def process_results(results, output_path):
        trainset_correctness = np.vstack([ret['trainset_correctness'] for ret in results])  # n_runs x train_size
        trainset_predictions = np.vstack([ret['trainset_predictions'] for ret in results])  # n_runs x train_size
        testset_correctness = np.vstack([ret['testset_correctness'] for ret in results])  # n_runs x test_size
        testset_predictions = np.vstack([ret['testset_predictions'] for ret in results])  # n_runs x test_size

        total_runs = trainset_correctness.shape[0]

        print(f'Avg test acc = {np.mean(testset_correctness):.4f}')
        print(f'Total runs = {total_runs}')
        
        def _masked_avg(x, mask, axis=0, esp=1e-10):
            return (np.sum(x * mask, axis=axis) / np.maximum(np.sum(mask, axis=axis), esp)).astype(np.float32)
        
        def _masked_dot(x, mask, esp=1e-10):
            x = x.T.astype(np.float32)
            return (np.matmul(x, mask) / np.maximum(np.sum(mask, axis=0, keepdims=True), esp)).astype(np.float32)
        
        # trainset_mask is a boolean mask indicating which data points are used in training
        trainset_mask = np.vstack([ret['trainset_mask'] for ret in results])
        inv_mask = np.logical_not(trainset_mask)

        mem_incl_p = _masked_avg(trainset_correctness, trainset_mask)
        mem_excl_p = _masked_avg(trainset_correctness, inv_mask)
        
        mem_est = _masked_avg(trainset_correctness, trainset_mask) - _masked_avg(trainset_correctness, inv_mask)
        infl_est = _masked_dot(testset_correctness, trainset_mask) - _masked_dot(testset_correctness, inv_mask)
        
        results_dict = dict(total_runs=total_runs,
                            trainset_mask=trainset_mask,
                            trainset_correctness=trainset_correctness, 
                            trainset_predictions=trainset_predictions,
                            testset_correctness=testset_correctness,
                            testset_predictions=testset_predictions,
                            memorization=mem_est, 
                            influence=infl_est,
                            memorization_inclusion_prob=mem_incl_p,
                            memorization_exclusion_prob=mem_excl_p)
        
        np.savez(output_path, **results_dict)
        return results_dict
    
    if is_human:
        output_path = os.path.join(results_dir, 'cifar-rand1-human-infl-mem.npz')
    else:
        output_path = os.path.join(results_dir, 'cifar-rand1-syn-infl-mem.npz')

    aggregated_results = process_results(aggregated_results, output_path)

    return aggregated_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .npz results files.")
    parser.add_argument("results_dir", type=str, help="Directory containing the .npz results files.")
    parser.add_argument("--is_human", action="store_true", help="Flag indicating whether the results are human-related.")

    args = parser.parse_args()

    aggregate_results(args.results_dir, args.is_human)