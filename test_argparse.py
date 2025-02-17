import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, required=True)
parser.add_argument('--type', type=int)#nargs='+')

args = parser.parse_args()

def main():
    do_code(args)
    return

def do_code(args):
    print(f'Hello world! Hello {args.name}!')
    print(args.type, type(args.type))

main()