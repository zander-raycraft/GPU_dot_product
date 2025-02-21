import numpy as np
from logic_gates.control import Control
from logic_gates.multiplexer import Multiplexer
from logic_gates.ripple_adder import RippleAdder

class ALU:
    '''
        @breif: This class represents an Arithmetic Logic Unit (ALU). It receives binary inputs, performs
                computations, and returns the results in binary format before converting them back to real numbers.

        @params: num_bits -> int, the number of bits the ALU operates on
    '''
    def __init__(self, num_bits: int):
        self.num_bits = num_bits
        self.control_unit = Control()
        self.multiplexer = Multiplexer(7)  # 7 operations
        self.ripple_adder = RippleAdder(num_bits)

    '''
        @breif: Converts a real number to its binary representation.

        @params: num -> int, the decimal number to convert
        @returns: binary_list -> list[int], the binary representation (LSB to MSB)
    '''
    def to_binary(self, num: int):
        return list(map(int, format(num, f'0{self.num_bits}b')))

    '''
        @breif: Converts a binary number (list) back to a real number.

        @params: binary_list -> list[int], the binary number (LSB to MSB)
        @returns: num -> int, the decimal equivalent
    '''
    def from_binary(self, binary_list: list):
        return int("".join(map(str, binary_list)), 2)

    '''
        @breif: Executes an ALU operation on two decimal numbers.

        @params: A -> int, first decimal number
        @params: B -> int, second decimal number
        @params: control_code -> str, a 3-bit binary string representing the operation

        @returns: result -> int, computed result as a decimal number
    '''
    def execute(self, A: int, B: int, control_code: str):
        A_bin = self.to_binary(A)
        B_bin = self.to_binary(B)

        # Decode control code
        operation = self.control_unit.decode(control_code)

        # Perform operations in binary
        if operation == "ADD":
            result_bin, _ = self.ripple_adder.add(A_bin, B_bin)
        elif operation == "SUB":
            B_twos_complement = self.twos_complement(B_bin)
            result_bin, _ = self.ripple_adder.add(A_bin, B_twos_complement)
        elif operation == "AND":
            result_bin = [a & b for a, b in zip(A_bin, B_bin)]
        elif operation == "OR":
            result_bin = [a | b for a, b in zip(A_bin, B_bin)]
        elif operation == "XOR":
            result_bin = [a ^ b for a, b in zip(A_bin, B_bin)]
        elif operation == "MUL":
            result_bin = self.to_binary(self.from_binary(A_bin) * self.from_binary(B_bin))
        elif operation == "DIV":
            result_bin = self.binary_divide(A_bin, B_bin)
        else:
            result_bin = [0] * self.num_bits 

        # Convert binary result back to decimal
        return self.from_binary(result_bin)

    '''
        @breif: Computes the two’s complement of a binary number.

        @params: B_bin -> list[int], binary number represented as a list of bits (LSB to MSB)
        @returns: twos_complement_B -> list[int], two’s complement representation
    '''
    def twos_complement(self, B_bin: list):
        inverted_B = [1 - bit for bit in B_bin]  # Flip bits
        one = [1] + [0] * (self.num_bits - 1)
        result, _ = self.ripple_adder.add(inverted_B, one)
        return result

    '''
        @breif: Performs binary multiplication using iterative addition.

        @params: A_bin -> list[int], first binary number (LSB to MSB)
        @params: B_bin -> list[int], second binary number (LSB to MSB)
        @returns: product_bin -> list[int], binary multiplication result (size num_bits)
    '''
    def binary_multiply(self, A_bin: list, B_bin: list):
        A_dec = self.from_binary(A_bin)
        B_dec = self.from_binary(B_bin)
        product_dec = A_dec * B_dec
        return self.to_binary(product_dec)

    '''
        @breif: Performs binary division using repeated subtraction.

        @params: A_bin -> list[int], numerator in binary (LSB to MSB)
        @params: B_bin -> list[int], denominator in binary (LSB to MSB)
        @returns: quotient_bin -> list[int], binary quotient (size num_bits)
    '''
    def binary_divide(self, A_bin: list, B_bin: list):
        A_dec = self.from_binary(A_bin)
        B_dec = self.from_binary(B_bin)
        if B_dec == 0:
            return self.to_binary(0)  # Prevent division by zero
        quotient_dec = A_dec // B_dec
        return self.to_binary(quotient_dec)
