import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, required=True)
parser.add_argument('--type', type=int)#nargs='+')

args = parser.parse_args()
name = args.name
t = args.type

print(f'Hello world! Hello {name}!')

print(t, type(t))