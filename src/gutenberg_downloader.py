"""
Project Gutenberg book downloader.

Downloads books from Project Gutenberg's top books list and saves them
as text files. Handles errors gracefully and provides progress feedback.
"""

import os
import re
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from string import Template
import requests
from bs4 import BeautifulSoup


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GutenbergDownloader:
    """
    Downloads books from Project Gutenberg.
    
    Handles scraping the top books list, extracting book metadata,
    downloading text files, and saving them with clean filenames.
    """
    
    # Base URL templates
    TOP_BOOKS_URL = 'http://www.gutenberg.org/browse/scores/top'
    BOOK_FILE_URL_TEMPLATE = Template('http://www.gutenberg.org/files/$book_id/$book_id.txt')
    
    # Request settings
    REQUEST_TIMEOUT = 30  # seconds
    REQUEST_DELAY = 1.0   # seconds between requests (be polite to server)
    
    def __init__(self, download_dir: str = 'raw_data/gutenberg'):
        """
        Initialize downloader.
        
        Args:
            download_dir: Directory to save downloaded books
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Tool)'
        })
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse HTML page.
        
        Args:
            url: URL to fetch
        
        Returns:
            BeautifulSoup object or None if request fails
        """
        try:
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
    
    def _extract_book_info(self, a_tag) -> Optional[Tuple[str, str]]:
        """
        Extract book name and ID from anchor tag.
        
        Args:
            a_tag: BeautifulSoup anchor tag element
        
        Returns:
            Tuple of (book_name, book_id) or None if extraction fails
        """
        try:
            # Extract book name: "Title (Author)" -> "Title"
            text_match = re.match(r'(.*?)(?:\s*\(\d+\))?$', a_tag.text.strip())
            if not text_match:
                return None
            book_name = text_match.group(1).strip()
            
            # Extract book ID from href: "/ebooks/12345" -> "12345"
            href = a_tag.get('href', '')
            id_match = re.match(r'/ebooks/(\d+)', href)
            if not id_match:
                return None
            book_id = id_match.group(1)
            
            return book_name, book_id
        except (AttributeError, IndexError) as e:
            logger.warning(f"Failed to extract book info from tag: {e}")
            return None
    
    def _clean_filename(self, filename: str) -> str:
        """
        Clean filename by removing invalid characters.
        
        Args:
            filename: Original filename
        
        Returns:
            Cleaned filename safe for filesystem
        """
        # Remove invalid characters for filenames
        invalid_chars = r'[<>:"/\\|?*]'
        cleaned = re.sub(invalid_chars, '_', filename)
        
        # Remove leading/trailing spaces and dots
        cleaned = cleaned.strip(' .')
        
        # Limit length
        max_length = 200
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length]
        
        # Ensure not empty
        if not cleaned:
            cleaned = 'untitled'
        
        return cleaned
    
    def _download_book(self, book_id: str, book_name: str) -> bool:
        """
        Download a single book from Project Gutenberg.
        
        Args:
            book_id: Project Gutenberg book ID
            book_name: Book name for filename
        
        Returns:
            True if download successful, False otherwise
        """
        # Construct URL
        url = self.BOOK_FILE_URL_TEMPLATE.substitute(book_id=book_id)
        
        # Create filename
        filename = self._clean_filename(book_name)
        filepath = self.download_dir / f"{filename}.txt"
        
        # Skip if already exists
        if filepath.exists():
            logger.info(f"Skipping {book_name} (already exists)")
            return True
        
        try:
            # Download book
            logger.info(f"Downloading: {book_name} (ID: {book_id})")
            response = self.session.get(url, timeout=self.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            logger.info(f"Saved: {filepath}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to download {book_name} (ID: {book_id}): {e}")
            return False
        except IOError as e:
            logger.error(f"Failed to save {book_name}: {e}")
            return False
    
    def get_top_books(self, k: int = 100) -> List[Tuple[str, str]]:
        """
        Get list of top K books from Project Gutenberg.
        
        Args:
            k: Number of top books to retrieve
        
        Returns:
            List of (book_name, book_id) tuples
        """
        logger.info(f"Fetching top {k} books from Project Gutenberg...")
        
        soup = self._fetch_page(self.TOP_BOOKS_URL)
        if not soup:
            return []
        
        # Find the "Last 30 days" section
        books_section = soup.find(id='books-last30')
        if not books_section:
            logger.error("Could not find 'books-last30' section on page")
            return []
        
        # Find the ordered list (more robust than next_sibling.next_sibling)
        ol = books_section.find_next('ol')
        if not ol:
            logger.error("Could not find book list on page")
            return []
        
        # Extract book information
        books = []
        for a_tag in ol.find_all('a', href=re.compile(r'/ebooks/\d+')):
            book_info = self._extract_book_info(a_tag)
            if book_info:
                books.append(book_info)
                if len(books) >= k:
                    break
        
        logger.info(f"Found {len(books)} books")
        return books
    
    def _request_user_approval(self, k: int) -> bool:
        """
        Request explicit user approval for downloading more than 3 books.
        
        Project Gutenberg's terms of service prohibit automated/bot access.
        For downloads > 3 books, we require explicit user approval.
        
        Args:
            k: Number of books requested
        
        Returns:
            True if user approves, False otherwise
        """
        print("\n" + "=" * 70)
        print("WARNING: Project Gutenberg Bot Policy")
        print("=" * 70)
        print("Project Gutenberg's terms of service prohibit the use of")
        print("automated tools, bots, or scripts to download multiple files.")
        print()
        print(f"You are requesting to download {k} books.")
        print("This may violate Project Gutenberg's terms of service.")
        print()
        print("Please ensure you are:")
        print("  - Downloading for personal/educational use")
        print("  - Respecting rate limits and server resources")
        print("  - Not using this for bulk/commercial purposes")
        print()
        print("=" * 70)
        
        while True:
            response = input(f"Do you want to proceed with downloading {k} books? (yes/no): ").strip().lower()
            if response in ('yes', 'y'):
                print("Proceeding with download...")
                return True
            elif response in ('no', 'n'):
                print("Download cancelled by user.")
                return False
            else:
                print("Please enter 'yes' or 'no'.")
    
    def download_top_k(self, k: int = 2, require_approval: bool = True) -> int:
        """
        Download top K books from Project Gutenberg.
        
        Note: For k > 3, this function will request explicit user approval
        due to Project Gutenberg's terms of service prohibiting automated
        bot access. The user must explicitly confirm before proceeding.
        
        Args:
            k: Number of top books to download
            require_approval: If True, request user approval for k > 3.
                             If False, skip approval (not recommended).
        
        Returns:
            Number of successfully downloaded books (0 if user cancels)
        """
        if k <= 0:
            logger.warning(f"Invalid k={k}, must be positive")
            return 0
        
        # Request user approval for downloads > 3 books
        if require_approval and k > 3:
            if not self._request_user_approval(k):
                logger.info("Download cancelled by user")
                return 0
        
        # Get list of top books
        books = self.get_top_books(k)
        if not books:
            logger.error("No books found to download")
            return 0
        
        # Download each book
        logger.info(f"Starting download of {len(books)} books...")
        success_count = 0
        
        for i, (book_name, book_id) in enumerate(books, 1):
            logger.info(f"[{i}/{len(books)}] Processing: {book_name}")
            
            if self._download_book(book_id, book_name):
                success_count += 1
            
            # Be polite: delay between requests
            if i < len(books):
                time.sleep(self.REQUEST_DELAY)
        
        logger.info(f"Download complete: {success_count}/{len(books)} books downloaded")
        return success_count


def download_top_k(k: int = 2, download_dir: str = 'raw_data/gutenberg', 
                   require_approval: bool = True) -> int:
    """
    Convenience function to download top K books from Project Gutenberg.
    
    Note: For k > 3, this function will request explicit user approval
    due to Project Gutenberg's terms of service prohibiting automated
    bot access. The user must explicitly confirm before proceeding.
    
    Args:
        k: Number of top books to download
        download_dir: Directory to save downloaded books
        require_approval: If True, request user approval for k > 3.
                         If False, skip approval (not recommended).
    
    Returns:
        Number of successfully downloaded books (0 if user cancels)
    
    Example:
        >>> download_top_k(k=5)
        # Will prompt for approval if k > 3
        Downloading top 5 books from Project Gutenberg...
    """
    downloader = GutenbergDownloader(download_dir=download_dir)
    return downloader.download_top_k(k, require_approval=require_approval)


if __name__ == '__main__':
    # Example usage
    download_top_k(k=5)

