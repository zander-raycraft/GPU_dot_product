import numpy as np
from gpu.warp import Warp

class StreamMulti:
    '''
        @breif: This is the streaming multiprocessor (SM), responsible for distributing workloads to Warps,
                executing computations in parallel across multiple ALUs, and storing results in local memory.

        @params: num_warps -> int, the number of warps in this SM.
        @params: num_threads_per_warp -> int, the number of ALU threads per warp.
        @params: num_bits -> int, the bit precision for ALU computations.
        @params: local_mem -> np.array, local storage for intermediate ALU results.
        @params: warps -> list[Warp], the warps executing computations.
    '''
    def __init__(self, num_warps: int, num_threads_per_warp: int, num_bits: int):
        self.num_warps = num_warps
        self.num_threads_per_warp = num_threads_per_warp
        self.num_bits = num_bits
        self.warps = [Warp(num_threads_per_warp, num_bits) for _ in range(num_warps)]
        self.local_mem = np.zeros((num_warps, num_threads_per_warp), dtype=np.int32)

    '''
        @breif: Distributes data across the warps inside this SM.

        @params: data_chunk -> np.array, the subsection of the input matrix assigned to this SM.
        @returns: None
    '''
    def distribute_data(self, data_chunk: np.array):
        # Remove strict size assertion; handle variable chunk sizes
        total_threads = self.num_warps * self.num_threads_per_warp
        chunk_size = data_chunk.shape[0]

        if chunk_size > total_threads:
            # Truncate excess data if chunk is too large
            data_chunk = data_chunk[:total_threads]
        elif chunk_size < total_threads:
            # Pad with zeros if chunk is too small
            padding = np.zeros((total_threads - chunk_size, 2), dtype=np.int32)
            data_chunk = np.vstack((data_chunk, padding))

        # Reshape data for warps
        reshaped_data = data_chunk.reshape(self.num_warps, self.num_threads_per_warp, 2)

        # Assign each warp its subset of data
        for i, warp in enumerate(self.warps):
            warp_data = list(map(tuple, reshaped_data[i]))
            warp.distribute_work(warp_data)

    '''
        @breif: Runs computations in parallel on all Warps.

        @params: control_code -> str, a 3-bit binary string specifying the operation.
        @returns: None
    '''
    def run_calculations(self, control_code: str):
        for i, warp in enumerate(self.warps):
            warp.run(control_code)

    '''
        @breif: Collects computed results from all warps and stores them in local memory.

        @returns: np.array, the computed results from all ALUs in this SM.
    '''
    def gather_results(self) -> np.array:
        for i, warp in enumerate(self.warps):
            self.local_mem[i] = warp.gather_results()
        return self.local_mem.flatten()

    '''
        @breif: Sends data from local memory in the SM back to global GPU memory.

        @returns: np.array, the final processed data to be sent to the GPU_SIM.
    '''
    def send_info(self) -> np.array:
        return self.gather_results()