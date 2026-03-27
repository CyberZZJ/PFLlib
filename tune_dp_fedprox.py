#!/usr/bin/env python
import subprocess
import json
import time
import os
import numpy as np

# Define the search space
search_space = {
    'mu': [0.001, 0.01, 0.1, 1.0],
    'dp_clip_norm': [0.5, 1.0, 2.0, 5.0],
    'dp_epsilon': [0.5, 1.0]
}

# Default parameters
default_params = {
    'algorithm': 'DPProx',
    'dataset': 'MNIST',
    'model': 'CNN',
    'batch_size': 10,
    'local_learning_rate': 0.005,
    'global_rounds': 100,
    'local_epochs': 1,
    'join_ratio': 1.0,
    'num_clients': 20,
    'times': 1,
    'eval_gap': 10,
    'device': 'cuda'
}

# Results storage
results = []

# Create results directory if it doesn't exist
results_dir = 'tuning_results'
os.makedirs(results_dir, exist_ok=True)

def run_experiment(params):
    """Run a single experiment with given parameters"""
    # Build command
    cmd = ['python', 'system/main.py']
    
    # Add default parameters
    for key, value in default_params.items():
        cmd.extend([f'--{key}', str(value)])
    
    # Add tuning parameters
    cmd.extend(['--mu', str(params['mu'])])
    cmd.extend(['--dp_clip', str(params['dp_clip_norm'])])
    cmd.extend(['--dp_eps', str(params['dp_epsilon'])])
    
    # Add unique save folder name
    save_folder = f"dp_prox_tune_mu{params['mu']}_clip{params['dp_clip_norm']}_eps{params['dp_epsilon']}"
    cmd.extend(['--save_folder_name', save_folder])
    
    print(f"Running experiment with mu={params['mu']}, clip={params['dp_clip_norm']}, eps={params['dp_epsilon']}")
    
    # Run the command
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    
    print(f"Experiment completed in {elapsed_time:.2f} seconds")
    
    # Parse results
    # Note: This is a simplified parsing. In practice, you might need to read the saved results files
    # For now, we'll extract accuracy from the output
    accuracy = 0.0
    for line in result.stdout.split('\n'):
        if 'Accuracy' in line:
            try:
                accuracy = float(line.split(':')[-1].strip())
            except:
                pass
    
    # Calculate convergence speed (simplified as accuracy per unit time)
    convergence_speed = accuracy / elapsed_time if elapsed_time > 0 else 0
    
    # Create result entry
    result_entry = {
        'params': params,
        'accuracy': accuracy,
        'elapsed_time': elapsed_time,
        'convergence_speed': convergence_speed
    }
    
    return result_entry

# Perform grid search
total_experiments = len(search_space['mu']) * len(search_space['dp_clip_norm']) * len(search_space['dp_epsilon'])
current_experiment = 0

for mu in search_space['mu']:
    for clip_norm in search_space['dp_clip_norm']:
        for epsilon in search_space['dp_epsilon']:
            current_experiment += 1
            print(f"\nExperiment {current_experiment}/{total_experiments}")
            
            params = {
                'mu': mu,
                'dp_clip_norm': clip_norm,
                'dp_epsilon': epsilon
            }
            
            result = run_experiment(params)
            results.append(result)
            
            # Save intermediate results
            with open(f'{results_dir}/intermediate_results.json', 'w') as f:
                json.dump(results, f, indent=2)

# Analyze results
def select_optimal_parameters(results):
    """Select optimal parameters based on accuracy, convergence, and computational efficiency"""
    if not results:
        return None
    
    # Calculate scores
    for result in results:
        # Normalize metrics
        accuracy = result['accuracy']
        time_efficiency = 1.0 / result['elapsed_time'] if result['elapsed_time'] > 0 else 0
        convergence = result['convergence_speed']
        
        # Weighted score (you can adjust weights based on priority)
        result['score'] = 0.5 * accuracy + 0.25 * time_efficiency + 0.25 * convergence
    
    # Sort by score
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    return sorted_results[0]

# Find optimal parameters
optimal_result = select_optimal_parameters(results)

# Save final results
with open(f'{results_dir}/tuning_results.json', 'w') as f:
    json.dump({
        'all_results': results,
        'optimal_result': optimal_result
    }, f, indent=2)

# Print summary
print("\n" + "="*80)
print("TUNING SUMMARY")
print("="*80)

print("\nOptimal Parameters:")
if optimal_result:
    print(f"mu: {optimal_result['params']['mu']}")
    print(f"Clip Norm: {optimal_result['params']['dp_clip_norm']}")
    print(f"Epsilon: {optimal_result['params']['dp_epsilon']}")
    print(f"Accuracy: {optimal_result['accuracy']:.4f}")
    print(f"Elapsed Time: {optimal_result['elapsed_time']:.2f}s")
    print(f"Convergence Speed: {optimal_result['convergence_speed']:.6f}")
    print(f"Score: {optimal_result['score']:.6f}")
else:
    print("No results found.")

print("\nAll Results:")
for i, result in enumerate(results):
    print(f"\nExperiment {i+1}:")
    print(f"  Parameters: mu={result['params']['mu']}, clip={result['params']['dp_clip_norm']}, eps={result['params']['dp_epsilon']}")
    print(f"  Accuracy: {result['accuracy']:.4f}")
    print(f"  Time: {result['elapsed_time']:.2f}s")
    print(f"  Convergence: {result['convergence_speed']:.6f}")
    if 'score' in result:
        print(f"  Score: {result['score']:.6f}")

print("\n" + "="*80)
print("Tuning completed. Results saved to tuning_results/")
print("="*80)