from typing import List
from logic_gates.alu import ALU

class Warp:
    '''
        @breif: This class represents a warp, which is a collection of threads that execute computations
                in parallel using ALUs. Each thread in the warp processes a piece of data independently.

        @params: num_threads -> int, the number of threads (ALUs) within the warp.
        @params: num_bits -> int, the number of bits each ALU operates on.
        @params: alus -> list[ALU], a list of ALU instances corresponding to each thread.
        @params: local_memory -> list[int], storage for intermediate results before returning to the SM.
    '''
    def __init__(self, num_threads: int, num_bits: int):
        self.num_threads = num_threads
        self.num_bits = num_bits
        self.alus = [ALU(num_bits) for _ in range(num_threads)]
        self.local_memory = [None] * num_threads

    '''
        @breif: Distributes work across the warp's ALUs. Each ALU processes a separate data pair.

        @params: work_data -> List[tuple[int, int]], a list of (A, B) tuples representing binary inputs.
        @returns: None
    '''
    def distribute_work(self, work_data: List[tuple[int, int]]):
        assert len(work_data) == self.num_threads, "Work data must match the number of threads."
        self.work_data = work_data

    '''
        @breif: Executes computations in parallel on all ALUs within the warp.

        @params: control_code -> str, a 3-bit binary string specifying the operation.
        @returns: None
    '''
    def run(self, control_code: str):
        for i in range(self.num_threads):
            A, B = self.work_data[i]

            # Execute ALU operation and store result
            result = self.alus[i].execute(A, B, control_code)
            self.local_memory[i] = result

    '''
        @breif: Collects the results stored in the local memory and returns them to the SM.

        @returns: List[int], the computed results from all threads.
    '''
    def gather_results(self) -> List[int]:
        return self.local_memory
