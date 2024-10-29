import argparse
import os
import pandas as pd
from pathlib import Path


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Materialize dataset files.')

    # Add arguments
    parser.add_argument('--output_dir', type=str, help='output directory where dataset files will be saved.', default=os.getcwd())
    parser.add_argument('--name', type=str, help='name of the dataset', default='penguins')

    # Parse arguments
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    penguins = pd.read_csv(script_dir / '..' / 'data' / 'dataset.csv')

    # Write to disk
    output_path = Path(args.output_dir) / f'{args.name}.csv'
    penguins.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
