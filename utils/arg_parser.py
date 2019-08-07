
def parse_input_args(command_line_args):

    if len(command_line_args) > 1:
        raise ValueError("Too many input arguments provided")

    if len(command_line_args) == 0:
        return 0

    task = command_line_args[0]

    return task

