"""
Investor Day Scout: Automated PR Parsing Pipeline
Bypasses Wall Street calendar paywalls using Free Google News RSS and Regex.
"""

import json
import logging
import re
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def update_investor_day_calendar(file_path: str = "investor_days.json"):
    """
    Scrapes Google News RSS for recent Investor Day announcements.
    Parses the Ticker and Date via NLP Regex, and updates the local JSON database.
    """
    # Exact phrase matching to filter out post-event recaps and noise
    query = '"to+host+investor+day"+OR+"to+host+analyst+day"'
    url = f'https://news.google.com/rss/search?q={query}+when:14d&hl=en-US&gl=US&ceid=US:en'
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=10)
        root = ET.fromstring(response.read())
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed: {e}")
        return

    # Load existing database (or create a blank dictionary)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                calendar = json.load(f)
            except json.JSONDecodeError:
                calendar = {}
    else:
        calendar = {}

    # Standard PR Ticker Format: (NASDAQ: AAPL) or NYSE: CRM
    ticker_pattern = re.compile(r'\b(?:NASDAQ|NYSE|nasdaq|nyse):\s*([A-Za-z]+)\b')
    
    # Matches textual date formats like: November 15, Oct 24, September 3
    date_pattern = re.compile(
        r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})', 
        re.IGNORECASE
    )

    current_year = datetime.now().year
    new_events_found = 0

    # Parse the XML feed
    for item in root.findall('.//item'):
        title = item.find('title').text
        desc = item.find('description').text if item.find('description') is not None else ""
        text_to_search = f"{title} {desc}"
        
        ticker_match = ticker_pattern.search(text_to_search)
        date_match = date_pattern.search(title) # Dates are usually in the headline
        
        if ticker_match and date_match:
            ticker = ticker_match.group(1).upper()
            month_str, day_str = date_match.groups()
            
            try:
                # Convert text date (e.g., "Nov 15") into quant-ready "YYYY-MM-DD"
                clean_month = month_str[:3].capitalize()
                date_obj = datetime.strptime(f"{clean_month} {day_str} {current_year}", "%b %d %Y")
                
                # If the parsed date is in the past, assume the PR is announcing next year's event
                if date_obj < datetime.now() - timedelta(days=30):
                    date_obj = date_obj.replace(year=current_year + 1)
                    
                formatted_date = date_obj.strftime("%Y-%m-%d")
                
                # Only add if it's a valid future date
                if datetime.strptime(formatted_date, "%Y-%m-%d") >= datetime.now():
                    if calendar.get(ticker) != formatted_date:
                        calendar[ticker] = formatted_date
                        new_events_found += 1
                        logger.info(f"🎯 Discovered New Investor Day: {ticker} on {formatted_date}")
                        
            except Exception:
                pass

    # Self-Cleaning Mechanism: Remove events that occurred more than 2 days ago
    cleaned_calendar = {}
    for tkr, evt_date in calendar.items():
        try:
            if datetime.strptime(evt_date, "%Y-%m-%d") >= datetime.now() - timedelta(days=2):
                cleaned_calendar[tkr] = evt_date
        except Exception:
            pass

    # Save the updated database back to the JSON file
    with open(file_path, "w") as f:
        json.dump(cleaned_calendar, f, indent=4)
        
    logger.info(f"Database updated. {new_events_found} new events injected. Total tracked: {len(cleaned_calendar)}")

if __name__ == "__main__":
    update_investor_day_calendar()
