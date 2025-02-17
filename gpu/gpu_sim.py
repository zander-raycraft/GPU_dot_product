import numpy as np
from sm import stream_multi

'''
    @breif: This is the GPU_SIM class, this class is the actual instance that runs in the instance.py, and is
            meant to (at a high level) mimic GPU structure to work with data. You can define it to have as many
            streaming multiprocessors, "sm", as you would like however it is default with two. This will take in an
            np.array and then tile it out, or break it into its parts, and send each tile to an sm and that sm
            will run its warp, "group of threads", in parallel on the ALUs for computation based on what is sent
            as the control units
            
    @params: num_sms -> int, this represents the number of straming multiprocessors will be in the GPU sim
    @params: mem_size -> int, this represents the size of the global memory for the GPU where all the values from
                        the ALUs output will be stored

'''
class GPU_SIM():
    
    '''
        @params: this is the initalization function, it initialilzes a GPU_SIM instance and its variables
    '''
    def __init__(self, num_sms = 2, mem_size = 1024):
        self.global_memory = np.zeros(mem_size, dtype=np.int32)
        self.sm_list = [None for _ in num_sms]
        self.arr1 = np.array
        self.arr2 = np.array
    
    '''
        @breif: this function takes in the input arrays to have operations done on and places them withing the
                GPU_SIM's interal vars
                
        @params: arr1 -> np.array, this represents the first array to be operated on
        @params: arr2 -> np.array, this represents the secind array to be operated on
        @returns: none
    
    '''
    def load_info(x: np.array, y: np.array):
        pass
    
    
    '''
        @breif: this funciton distributes the data across all of the streaming multiprocessers
    '''
    def distribute_data(self):
        pass
    
    
    '''
        @breif: this function runs the computation across the streaming multiprocessors and is essentially the
                control function for them sending them into active states
                
        @params: control_code -> int, this is the code that tells the SMs what opearation they need to do for
                                the data loaded into the SMs, this control code is mapped to operations [0-3]
                                which then gets translated to be the control bits for the ALU computation
    '''
    def run_computation(self, control_code: int) -> list:
        pass
    
    '''
        @breif: This funciton pulls the information/data in the global memory where it reconstructs it and
                sends the GPU response back to the object that asked it to operate on the data
                
        @returns: result -> np.array, this is the final output from the GPU_SIM sent to the user
    '''
    def reconstruct_data(self) -> np.array:
        pass