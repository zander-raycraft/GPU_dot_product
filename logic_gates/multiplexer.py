class Multiplexer:
    '''
        @breif: This class simulates an N-to-1 multiplexer. A multiplexer selects one input from multiple
                inputs based on the value of the control signal.

        @params: num_inputs -> int, the number of input signals the multiplexer handles
    '''
    def __init__(self, num_inputs: int):
        self.num_inputs = num_inputs
        self.control_bits = num_inputs.bit_length() - 1

    '''
        @breif: This function selects one input based on the control signal.

        @params: inputs -> list[int], list of possible input values
        @params: control_signal -> list[int], binary list representing the control bits
        @returns: selected_output -> int, the selected input value
    '''
    def select(self, inputs: list, control_signal: list) -> int:
        # Ensure inputs are correct
        assert len(inputs) == self.num_inputs, "Number of inputs must match num_inputs"
        assert len(control_signal) == self.control_bits, "Control signal must have the correct bit length"

        # Convert control bits to an integer index
        selected_index = int("".join(map(str, control_signal)), 2)
        return inputs[selected_index]

