import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description='Small script to create sample from given dataset')
parser.add_argument('input_path', metavar='P', type=str, nargs='?',
                    help='path for dataset need to be sampled')
parser.add_argument('--n', metavar='N', type=int, nargs='?', default=1000,
                    help='number of data counts')
parser.add_argument('--output_path', metavar='O', type=str, nargs='?', default=None,
                    help='output path for new sample')
parser.add_argument('--choice', metavar='C', type=str, nargs='?', default='straight',
                    help='choose straight or random')



args = parser.parse_args()
if args.output_path is None:
    args.output_path = os.path.join('/'.join(args.input_path.split('/')[:-1]), #  drop filename
                                                   'sample_{}_{}.csv'.format(args.n, args.choice))

df = pd.read_csv(args.input_path)
if args.choice == 'straight':
    df.iloc[:args.n].to_csv(args.output_path, index=False)
if args.choice == 'random':
    df.sample(n=args.n).to_csv(args.output_path, index=False)
