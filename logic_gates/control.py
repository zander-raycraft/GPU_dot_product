class Control:
    '''
        @breif: This class serves as the control unit for the ALU. It receives a control code, decodes it, 
                and sends signals to the ALU to perform the corresponding operation.

        @params: control_map -> dict, a mapping of control codes to ALU operations
    '''
    def __init__(self):
        '''
            @breif: Initializes the control unit with predefined operation mappings.
        '''
        self.control_map = {
            "000": "ADD",      # Addition
            "001": "SUB",      # Subtraction
            "010": "AND",      # Bitwise AND
            "011": "OR",       # Bitwise OR
            "100": "XOR",      # Bitwise XOR
            "101": "MUL",      # Multiplication
            "110": "DIV",      # Division
            "111": "NOP"       # No operation
        }

    '''
        @breif: This function takes in a control code and deciphers the operation.

        @params: control_code -> str, a 3-bit binary string representing the operation
        @returns: operation -> str, the decoded operation name
    '''
    def decode(self, control_code: str) -> str:
        return self.control_map.get(control_code, "INVALID")