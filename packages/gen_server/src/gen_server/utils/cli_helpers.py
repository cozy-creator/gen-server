import sys


def find_subcommand():
    for arg in sys.argv[1:]:  # Skip the script name (sys.argv[0])
        if not arg.startswith("-"):
            return arg
    return None  # Return None if no subcommand is found


def find_arg_value(arg_name):
    for i, arg in enumerate(sys.argv):
        if arg.startswith(f"{arg_name}="):
            return arg.split("=", 1)[1]
        elif arg == arg_name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def parse_known_args_wrapper(parser, args=None, namespace=None):
    known_args, _ = parser.parse_known_args(args, namespace)
    return known_args
