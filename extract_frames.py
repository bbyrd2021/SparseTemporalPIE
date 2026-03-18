"""
Extract frames from PIE and JAAD video datasets.
Run this before training to prepare image data.
"""
import argparse
import os


def extract_pie(data_path):
    from utils.pie_data import PIE
    pie = PIE(data_path=data_path)
    print("Extracting PIE frames (annotated only)...")
    pie.extract_and_save_images(extract_frame_type='annotated')
    print("PIE frame extraction complete.")


def extract_jaad(data_path):
    from utils.jaad_data import JAAD
    jaad = JAAD(data_path=data_path)
    print("Extracting JAAD frames...")
    jaad.extract_and_save_images()
    print("JAAD frame extraction complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['pie', 'jaad', 'both'], default='both')
    parser.add_argument('--pie-path', type=str, default='/data/datasets/PIE')
    parser.add_argument('--jaad-path', type=str, default='/data/datasets/JAAD')
    args = parser.parse_args()

    if args.dataset in ('pie', 'both'):
        extract_pie(args.pie_path)
    if args.dataset in ('jaad', 'both'):
        extract_jaad(args.jaad_path)
