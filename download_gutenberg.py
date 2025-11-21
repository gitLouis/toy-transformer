"""
Example script to download books from Project Gutenberg.

Usage:
    python download_gutenberg.py --k 5
    python download_gutenberg.py --k 10 --dir raw_data/gutenberg
"""

import argparse
from src.gutenberg_downloader import download_top_k


def main():
    """Main entry point for Gutenberg downloader script."""
    parser = argparse.ArgumentParser(
        description='Download top books from Project Gutenberg'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of top books to download (default: 5)'
    )
    parser.add_argument(
        '--dir',
        type=str,
        default='raw_data/gutenberg',
        help='Directory to save downloaded books (default: raw_data/gutenberg)'
    )
    parser.add_argument(
        '--skip-approval',
        action='store_true',
        help='Skip user approval prompt (not recommended for k > 3)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Project Gutenberg Book Downloader")
    print("=" * 60)
    print(f"Downloading top {args.k} books...")
    print(f"Save directory: {args.dir}")
    print("-" * 60)
    
    require_approval = not args.skip_approval
    success_count = download_top_k(
        k=args.k, 
        download_dir=args.dir,
        require_approval=require_approval
    )
    
    print("-" * 60)
    print(f"Download complete: {success_count}/{args.k} books downloaded")
    print("=" * 60)


if __name__ == '__main__':
    main()

