from typing import List
from ..logic_gates.alu import ALU

class Warp:
    def __init__(self, num_threads: int):
        '''
            @brief: Initializes the warp with a specified number of threads.
            
            @params: num_threads -> int, the number of threads in this warp.
            @returns: None
        '''
        self.num_threads = num_threads
        self.threads = [None] * num_threads

    def distribute_work(self, work_data: List):
        '''
            @brief: Distributes the given work data among the warp's threads.
            
            @params: work_data -> List, a list of data chunks or tasks for each thread.
            @returns: None
        '''
        pass

    def run(self, control_code: int):
        '''
            @brief: Executes the tasks in the warp according to the provided control code.
            
            @params: control_code -> int, an integer (or Enum) that indicates which operation to perform.
            @returns: None
        '''
        pass

    def gather_results(self) -> List:
        '''
            @brief: Collects and returns the results from each thread in the warp.
            
            @returns: List, a list of results from the warp.
        '''
        pass
