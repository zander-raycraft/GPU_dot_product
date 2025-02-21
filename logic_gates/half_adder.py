class HalfAdder:
    '''
        @breif: This class works as the half-adder logic circuit, made up of an XOR gate and an AND gate.
                It is used as one of the building blocks of the full adder, which is just a half adder
                that feeds into another. These full adders are used in the ripple adder for the ALU.

        @params: A0 -> int, first input value (0 or 1)
        @params: A1 -> int, second input value (0 or 1)
    '''
    def __init__(self, A0: int, A1: int):
        self.A0 = A0
        self.A1 = A1

    '''
        @breif: This function computes the sum and carry of the half adder.

        @params: none
        @returns: sum_result -> int, computed as (A0 XOR A1)
        @returns: carry -> int, computed as (A0 AND A1)
    '''
    def compute(self):
        sum_result = self.A0 ^ self.A1 
        carry = self.A0 & self.A1
        return sum_result, carry
