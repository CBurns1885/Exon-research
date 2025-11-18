"""Web scrapers for central bank speeches and statements."""
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
from typing import List, Dict
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeechScraper:
    """Scrape central bank speeches from official websites."""

    def __init__(self):
        """Initialize scraper."""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def scrape_fed_speeches(self, max_speeches: int = 20) -> List[Dict]:
        """Scrape Federal Reserve speeches."""
        speeches = []

        try:
            url = "https://www.federalreserve.gov/newsevents/speeches.htm"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find speech entries
            speech_rows = soup.find_all('div', class_='row eventlist__event')[:max_speeches]

            for row in speech_rows:
                try:
                    # Extract date
                    date_elem = row.find('time')
                    date_str = date_elem.get('datetime') if date_elem else None
                    date_obj = datetime.fromisoformat(date_str) if date_str else datetime.now()

                    # Extract title and link
                    title_elem = row.find('div', class_='col-xs-9').find('a')
                    title = title_elem.text.strip() if title_elem else "Untitled"
                    link = title_elem.get('href') if title_elem else None

                    # Full URL
                    if link and not link.startswith('http'):
                        link = f"https://www.federalreserve.gov{link}"

                    # Extract speaker
                    speaker_elem = row.find('p', class_='news__speaker')
                    speaker = speaker_elem.text.strip() if speaker_elem else "Unknown"

                    speeches.append({
                        "bank_code": "FED",
                        "date": date_obj.date(),
                        "title": title,
                        "speaker": speaker,
                        "url": link,
                    })

                except Exception as e:
                    logger.warning(f"Error parsing FED speech row: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping FED speeches: {e}")

        return speeches

    def scrape_ecb_speeches(self, max_speeches: int = 20) -> List[Dict]:
        """Scrape European Central Bank speeches."""
        speeches = []

        try:
            url = "https://www.ecb.europa.eu/press/key/html/downloads.en.xml"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'xml')

            items = soup.find_all('item')[:max_speeches]

            for item in items:
                try:
                    title = item.find('title').text.strip() if item.find('title') else "Untitled"
                    link = item.find('link').text.strip() if item.find('link') else None
                    pub_date = item.find('pubDate').text.strip() if item.find('pubDate') else None

                    # Parse date
                    date_obj = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S %z') if pub_date else datetime.now()

                    # Extract speaker from title (usually format: "Speaker name: Title")
                    speaker = "ECB Official"
                    if ':' in title:
                        potential_speaker = title.split(':')[0].strip()
                        if len(potential_speaker) < 50:  # Reasonable name length
                            speaker = potential_speaker

                    speeches.append({
                        "bank_code": "ECB",
                        "date": date_obj.date(),
                        "title": title,
                        "speaker": speaker,
                        "url": link,
                    })

                except Exception as e:
                    logger.warning(f"Error parsing ECB speech: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping ECB speeches: {e}")

        return speeches

    def scrape_boe_speeches(self, max_speeches: int = 20) -> List[Dict]:
        """Scrape Bank of England speeches."""
        speeches = []

        try:
            url = "https://www.bankofengland.co.uk/news/speeches"
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            speech_cards = soup.find_all('article', class_='search-result')[:max_speeches]

            for card in speech_cards:
                try:
                    title_elem = card.find('h3')
                    title = title_elem.text.strip() if title_elem else "Untitled"

                    link_elem = card.find('a', href=True)
                    link = link_elem['href'] if link_elem else None
                    if link and not link.startswith('http'):
                        link = f"https://www.bankofengland.co.uk{link}"

                    date_elem = card.find('time')
                    date_str = date_elem.get('datetime') if date_elem else None
                    date_obj = datetime.fromisoformat(date_str.split('T')[0]) if date_str else datetime.now()

                    # Speaker often in the title or subtitle
                    speaker = "BoE Official"

                    speeches.append({
                        "bank_code": "BOE",
                        "date": date_obj.date(),
                        "title": title,
                        "speaker": speaker,
                        "url": link,
                    })

                except Exception as e:
                    logger.warning(f"Error parsing BoE speech: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping BoE speeches: {e}")

        return speeches

    def scrape_all_banks(self, max_per_bank: int = 10) -> Dict[str, List[Dict]]:
        """Scrape speeches from all central banks."""
        all_speeches = {}

        logger.info("Scraping Federal Reserve...")
        all_speeches["FED"] = self.scrape_fed_speeches(max_per_bank)
        time.sleep(2)

        logger.info("Scraping European Central Bank...")
        all_speeches["ECB"] = self.scrape_ecb_speeches(max_per_bank)
        time.sleep(2)

        logger.info("Scraping Bank of England...")
        all_speeches["BOE"] = self.scrape_boe_speeches(max_per_bank)
        time.sleep(2)

        # For other banks, return empty lists (can be implemented later)
        for bank in ["BOJ", "BOC", "RBA", "SNB", "RBNZ"]:
            all_speeches[bank] = []

        return all_speeches

    def fetch_speech_content(self, url: str) -> str:
        """Fetch full text content of a speech."""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try different content selectors
            content = None

            # Fed speeches
            content_div = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')
            if content_div:
                paragraphs = content_div.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs])

            # ECB speeches
            if not content:
                content_div = soup.find('div', class_='ecb-pressContentBox')
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = '\n\n'.join([p.text.strip() for p in paragraphs])

            # BoE speeches
            if not content:
                content_div = soup.find('article')
                if content_div:
                    paragraphs = content_div.find_all('p')
                    content = '\n\n'.join([p.text.strip() for p in paragraphs])

            # Generic fallback
            if not content:
                paragraphs = soup.find_all('p')
                content = '\n\n'.join([p.text.strip() for p in paragraphs if len(p.text.strip()) > 50])

            return content if content else ""

        except Exception as e:
            logger.error(f"Error fetching speech content from {url}: {e}")
            return ""

    def save_speeches_to_db(self, speeches: Dict[str, List[Dict]], db_session):
        """Save scraped speeches to database."""
        from backend.data.database import CentralBankSpeech

        total_saved = 0

        for bank_code, bank_speeches in speeches.items():
            for speech in bank_speeches:
                # Check if already exists
                existing = db_session.query(CentralBankSpeech).filter(
                    CentralBankSpeech.bank_code == bank_code,
                    CentralBankSpeech.url == speech['url']
                ).first()

                if not existing:
                    # Fetch full content
                    if speech.get('url'):
                        logger.info(f"Fetching content for: {speech['title'][:50]}...")
                        speech['text_content'] = self.fetch_speech_content(speech['url'])
                        time.sleep(1)  # Rate limiting

                    new_speech = CentralBankSpeech(**speech)
                    db_session.add(new_speech)
                    total_saved += 1

        db_session.commit()
        return total_saved
