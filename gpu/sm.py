import numpy as np
from warp import warp

'''
    @breif: This is the streaming multiprocessor, this SM houses a warp which iterates over the defined number
            ALUs. This copmutes part of the overall GPU_SIM input
            
    @params: warp -> CLASS:warp(), this is the collection of threads that will run simple operations in parallel
    @params: ALU_LIST -> list, this is a list for tracking and running the ALUs in parallel
    @params: local_mem -> int, this is the local memory where the ALU outputs are stored   
    @params: cc -> int, control code for what operation to do on the ALU    

'''
class stream_multi():
    
    '''
        @param: this is the intialization funciton for the SM class
    '''
    def __init__(self, num_alu: int, cc: int):
        self.warp = warp()
        self.ALU_LIST = []
        self.local_mem = np.zeros(512, dtype=np.int32)
        self.cc = cc
    
    '''
        @breif: This funciton sends data from the the local memory in the ALU to the global memory in the GPU
        
        @return: result -> np.array, this is the info from the local memory int the ALU
    '''
    def send_info(self) -> np.array:
        pass
    
    '''
        @breif: this function runs the calculations on the SM which controls the threads (warp) that runs
                parallel simple operations on the ALUs
    '''
    def run_calculations(self):
        pass