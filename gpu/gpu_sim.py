import numpy as np


'''
    @breif: This is the GPU_SIM class, this class is the actual instance that runs in the instance.py, and is
            meant to (at a high level) mimic GPU structure to work with data. You can define it to have as many
            streaming multiprocessors, "sm", as you would like however it is default with two. This will take in an
            np.array and then tile it out, or break it into its parts, and send each tile to an sm and that sm
            will run its warp, "group of threads", in parallel on the ALUs for computation based on what is sent
            as the control units
            
    @params: arr1 -> np.array, this represents the first array to be operated on
    @params: arr2 -> np.array, this represents the secind array to be operated on
    @params: num_sms -> int, this represents the number of straming multiprocessors will be in the GPU sim
    @params: mem_size -> int, this represents the size of the global memory for the GPU where all the values from
                        the ALUs output will be stored

'''
class GPU_SIM():
    
    def __init__(self, num_sms = 2, mem_size = 1024):
        self.global_memory = np.zeros(mem_size, dtype=np.int32)
        self.sm_list = [None for _ in num_sms]
    
    def distribute_data(self, arr1: np.array, arr2: np.array):
        pass
    
    def run_computation(self, control_code: int) -> list:
        pass
    
    def compute_output(self) -> np.array:
        pass