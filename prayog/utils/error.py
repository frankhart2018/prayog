import sys


def throw(error_type, error_msg):
    print(f"\033[91m\033[1m{error_type}:\033[m {error_msg}")
    sys.exit()
