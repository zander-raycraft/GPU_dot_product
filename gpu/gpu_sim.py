import numpy as np
from gpu.sm import StreamMulti

'''
    @brief: GPU simulator supporting dot product and matrix multiplication with a simple GPU-like structure.
            Splits data across streaming multiprocessors (SMs) for parallel computation.

    @params: num_sms -> int, number of streaming multiprocessors.
    @params: mem_size -> int, size of global memory for storing results.
    @params: num_warps -> int, number of warps per SM.
    @params: num_threads_per_warp -> int, number of threads per warp.
    @params: num_bits -> int, bit precision for ALU operations.
'''
class GPU_SIM:
    
    def __init__(self, num_sms=2, mem_size=1024, num_warps=2, num_threads_per_warp=4, num_bits=8):
        self.global_memory = np.zeros(mem_size, dtype=np.int32)
        self.num_sms = num_sms
        self.num_warps = num_warps
        self.num_threads_per_warp = num_threads_per_warp
        self.num_bits = num_bits
        self.sm_list = [StreamMulti(num_warps, num_threads_per_warp, num_bits) for _ in range(num_sms)]
        self.arr1 = None
        self.arr2 = None
        self.arr1_flat = None
        self.arr2_flat = None
        self.operation = None
        self.output_shape = None
        self.task_map = []
        
    """
        @brief: Loads input arrays and determines operation type.

        @params: arr1 -> np.array, first input (vector or matrix).
        @params: arr2 -> np.array, second input (vector or matrix).
    """
    def load_info(self, arr1: np.array, arr2: np.array, operation="matmul"):
        self.arr1 = arr1
        self.arr2 = arr2
        self.operation = operation.lower()

        if self.operation == "dot":
            if arr1.ndim == 2 and arr2.ndim == 2:
                flat_arr1 = arr1.flatten()
                flat_arr2 = arr2.flatten()
                assert flat_arr1.shape == flat_arr2.shape, "Flattened vectors must have the same length for dot product."
                self.arr1_flat = flat_arr1
                self.arr2_flat = flat_arr2
            elif arr1.ndim == 1 and arr2.ndim == 1:
                assert arr1.shape == arr2.shape, "Vectors must have the same length for dot product."
                self.arr1_flat = arr1
                self.arr2_flat = arr2
            else:
                raise ValueError("For dot product, inputs must be 1D vectors or 2D matrices that can be flattened.")
            self.output_shape = (1,)
        elif self.operation == "matmul":
            assert arr1.ndim == 2 and arr2.ndim == 2, "Matrix multiplication requires 2D arrays."
            assert arr1.shape[1] == arr2.shape[0], "Matrix dimensions must match for multiplication (m x k) * (k x n)."
            self.output_shape = (arr1.shape[0], arr2.shape[1])
        else:
            raise ValueError("Operation must be 'dot' or 'matmul'.")

    """
        @brief: Distributes data across SMs based on operation type.
    """
    def distribute_data(self):
        assert self.arr1 is not None and self.arr2 is not None, "Input arrays must be loaded first."
        self.task_map = []

        if self.operation == "dot":
            flattened_data = np.column_stack((self.arr1_flat, self.arr2_flat))
            chunks = np.array_split(flattened_data, self.num_sms)
            for i, sm in enumerate(self.sm_list):
                sm.distribute_data(chunks[i])

        elif self.operation == "matmul":
            m, k = self.arr1.shape
            k, n = self.arr2.shape
            total_pairs = m * n * k  # (i, j, k) indexed multiplications
            pairs_per_sm = (total_pairs + self.num_sms - 1) // self.num_sms  # Ensure even workload

            pair_idx = 0
            for sm_idx, sm in enumerate(self.sm_list):
                data_pairs = []
                sm_task_map = []
                pairs_assigned = 0
                
                while pairs_assigned < pairs_per_sm and pair_idx < total_pairs:
                    task_idx = pair_idx // k
                    k_idx = pair_idx % k
                    i = task_idx // n
                    j = task_idx % n
                    data_pairs.append((self.arr1[i, k_idx], self.arr2[k_idx, j]))
                    sm_task_map.append((i, j, k_idx))
                    pairs_assigned += 1
                    pair_idx += 1
                
                data_array = np.array(data_pairs)
                expected_size = sm.num_warps * sm.num_threads_per_warp
                if len(data_array) < expected_size:
                    padding = np.zeros((expected_size - len(data_array), 2), dtype=int)
                    data_array = np.vstack((data_array, padding))
                    sm_task_map.extend([(0, 0, 0)] * (expected_size - len(data_pairs)))
                sm.distribute_data(data_array)
                self.task_map.extend(sm_task_map)

    """
        @brief: Runs multiplication across SMs and aggregates results.

        @params: control_code -> str, 3-bit binary string ("101" for MUL).
    """
    def run_computation(self, control_code: str):
        if control_code != "101":
            raise ValueError("Only multiplication ('101') is supported.")

        for sm in self.sm_list:
            sm.run_calculations(control_code)

        if self.operation == "dot":
            dot_product = sum(int(val) for sm in self.sm_list for val in sm.send_info())
            self.global_memory[0] = dot_product

        elif self.operation == "matmul":
            m, n = self.output_shape
            result = np.zeros((m, n), dtype=np.int32)
            all_results = np.concatenate([sm.send_info() for sm in self.sm_list])

            # Properly accumulate products for matrix multiplication
            sum_map = {}
            for idx, (i, j, k_idx) in enumerate(self.task_map):
                if idx < len(all_results):  # Ignore padding
                    if (i, j) not in sum_map:
                        sum_map[(i, j)] = 0
                    sum_map[(i, j)] += int(all_results[idx])  # Ensure int conversion

            for (i, j), val in sum_map.items():
                result[i, j] = val  # Assign final summed value per (i, j)

            self.global_memory[:m * n] = result.flatten()
            
    """
        @brief: Retrieves results from global memory.

        @returns: int (dot product) or np.array (matrix multiplication result).
    """
    def reconstruct_data(self):
        if self.operation == "dot":
            result = int(self.global_memory[0])
            return result
        elif self.operation == "matmul":
            m, n = self.output_shape
            result = self.global_memory[:m * n].reshape(m, n)
            return result
