import numpy as np
from gpu.gpu_sim import GPU_SIM

if __name__ == "__main__":
    arr1 = np.array([
        [13, 5, 7, 3],
        [12, 4, 9, 6],
        [8, 2, 14, 7],
        [11, 1, 10, 5]
    ])
    
    arr2 = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])

    gpu_matmul = GPU_SIM(num_sms=2, mem_size=1024, num_warps=4, num_threads_per_warp=8, num_bits=8)
    '''
        @code: operation = 'dot': this tells the GPU sim to run dot product of input data
        @code: operation = 'matmul': this tells the GPU sim to run matrix multiplication
        
        @NOTE: for larger matricies or higher order tensors, make sure you allocate the correct size for the
                GPU_SIM, it will run but will give incorrect truncated answers if not done to the correct size
    '''
    gpu_matmul.load_info(arr1, arr2, operation="matmul")
    gpu_matmul.distribute_data()
    gpu_matmul.run_computation("101")

    print('GPU matrix multiplication test:\n')
    matmul_result = gpu_matmul.reconstruct_data()
    numpy_matmul = np.matmul(arr1, arr2)
    print(f"GPU_SIM matrix multiplication:\n{matmul_result}")
    print(f"NumPy matrix multiplication:\n{numpy_matmul}\n")
    
    print('GPU dot product test:\n')
    gpu_dot = GPU_SIM(num_sms=2, mem_size=1024, num_warps=4, num_threads_per_warp=8, num_bits=8)
    gpu_dot.load_info(arr1, arr2, operation="dot")
    gpu_dot.distribute_data()
    gpu_dot.run_computation("101")
    print(f'GPU_SIM dot product: {gpu_dot.reconstruct_data()}')
    print(f'numpy dot product: {gpu_dot.reconstruct_data()}')
    
    
    
