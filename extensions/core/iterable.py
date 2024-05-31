# Similar to the random-seed + control-after-generate widget in ComfyUI.
# This has an initial value and then every time it produces a new value.
# This might be better implemented as a sub-graph (Macro) than a single
# custom node.

import random

class Iterable:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "current_value": ("INT", {"default": 0}),
                "next_value_method": (["fixed", "random", "increment", "decrement"],)
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_next_value"

    CATEGORY = "misc"

    def __init__(self):
        self.state = {}

    def generate_next_value(self, current_value, next_value_method="random"):
        if next_value_method == "fixed":
            next_value = current_value  # Keep the value the same
        elif next_value_method == "random":
            next_value = random.randint(0, 2**32 - 1)  # Generate a random integer
        elif next_value_method == "increment":
            next_value = current_value + 1  # Increment the value
        elif next_value_method == "decrement":
            next_value = current_value - 1  # Decrement the value
        else:
            raise ValueError(f"Invalid next_value_method: {next_value_method}")

        self.state['current_value'] = next_value  # Store the new value in the state

        return (next_value,)
    