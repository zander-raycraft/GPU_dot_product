from logic_gates.half_adder import HalfAdder

class RippleAdder:
    '''
        @breif: This class works as the ripple-carry adder, which performs multi-bit binary addition.
                It consists of multiple full adders, each of which adds two input bits and a carry bit.
                The carry-out from one full adder is passed as the carry-in to the next full adder.

        @params: num_bits -> int, the number of bits for the binary numbers to be added
    '''
    def __init__(self, num_bits: int):
        self.num_bits = num_bits

    '''
        @breif: This function performs binary addition using the ripple carry adder.
        
        @params: A -> list[int], first binary number represented as a list of bits (LSB to MSB)
        @params: B -> list[int], second binary number represented as a list of bits (LSB to MSB)
        @returns: sum_result -> list[int], the resulting sum as a binary list (LSB to MSB)
        @returns: final_carry -> int, the final carry-out bit after addition
    '''
    def add(self, A: list, B: list):
        assert len(A) == self.num_bits and len(B) == self.num_bits, "Input sizes must match num_bits"

        sum_result = []
        carry = 0  # Initial carry-in is 0

        for i in range(self.num_bits):
            # First half adder (adding input bits)
            ha1 = HalfAdder(A[i], B[i])
            sum1, carry1 = ha1.compute()

            # Second half adder (adding previous carry)
            ha2 = HalfAdder(sum1, carry)
            final_sum, carry2 = ha2.compute()

            # Store sum bit
            sum_result.append(final_sum)
            carry = carry1 | carry2 

        return sum_result, carry
