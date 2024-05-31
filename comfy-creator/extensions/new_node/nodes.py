# Has no inputs. Outputs a constant value.


CONSTANT_VALUE = 0  # The value to output

class ConstantNode:
    def __init__(self):
        pass

    RETURN_TYPES = ("*",)  
    FUNCTION = "pass_through"

    CATEGORY = "misc"

    def pass_through(self):
        return CONSTANT_VALUE  # Returns the constant value