import argparse
import os
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler


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

    # Prepare features
    features = penguins[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = features.dropna()
    features['sex'] = pd.factorize(features['sex'])[0]


    scaler = StandardScaler()
    numeric_features = features.drop(columns=['sex'])
    numeric_features_scaled = scaler.fit_transform(numeric_features)
    features_scaled_df = pd.DataFrame(numeric_features_scaled, columns=numeric_features.columns)
    features_scaled_df['sex'] = features['sex'].values

    # Prepare labels
    labels_df = penguins[['species']].copy()
    labels_df['species'] = pd.factorize(labels_df['species'])[0]
    labels_df.columns = ['label']

    # Write to disk
    features_scaled_df.to_csv(Path(args.output_dir) / f'{args.name}.features.csv', index_label='id')
    labels_df.to_csv(Path(args.output_dir) / f'{args.name}.labels.csv', index_label='id')


if __name__ == "__main__":
    main()
