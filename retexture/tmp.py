
import argparse
import sys

# Get arguments after '--'
args = sys.argv[sys.argv.index('--') + 1:]

parser = argparse.ArgumentParser()
parser.add_argument('--some_arg', help='Your custom argument')
args = parser.parse_args(args)

print(args.some_arg)


