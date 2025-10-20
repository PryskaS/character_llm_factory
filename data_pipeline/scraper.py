import requests
from bs4 import BeautifulSoup
import re

# The URL where the transcripts are hosted.
TRANSCRIPTS_URL = "https://rickandmorty.fandom.com/wiki/Category:Transcripts"

def get_transcript_links():
    """Fetches the main transcript page and extracts links to individual episode transcripts."""
    print("Fetching transcript category page...")
    try:
        response = requests.get(TRANSCRIPTS_URL)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.RequestException as e:
        print(f"Error fetching main page: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    links = []
    # Find all links within the main content area that point to episode transcripts.
    for a in soup.select("ul.category-page__members-for-char li.category-page__member a"):
        if a.has_attr('href'):
            full_url = "https://rickandmorty.fandom.com" + a['href']
            links.append(full_url)
    
    print(f"Found {len(links)} episode transcript links.")
    return links

def extract_rick_lines(url: str):
    """
    Extracts all of Rick's lines from a single episode transcript URL.
    """
    print(f"  Processing: {url.split('/')[-1]}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    - Could not fetch episode: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    lines = []
    # The transcripts use a pattern: a character's name is in a <b> tag, followed by their line.
    # Example: <p><b>Rick:</b> Wubba lubba dub dub!</p>
    for p in soup.find_all("p"):
        if p.b and "Rick:" in p.b.text:
            # Get the text of the line, excluding the "Rick:" part.
            line_text = p.get_text(separator=" ", strip=True).replace(p.b.text, "").strip()
            
            # Engineering Mindset: Data cleaning. Remove artifacts like scene descriptions in [brackets].
            line_text = re.sub(r'\[.*?\]', '', line_text)
            # Remove any leading colon or space left over.
            line_text = line_text.lstrip(': ').strip()

            if line_text:
                lines.append(line_text)
    
    return lines

def main():
    """Main function to orchestrate the scraping and data saving."""
    all_rick_lines = []
    links = get_transcript_links()

    for link in links:
        all_rick_lines.extend(extract_rick_lines(link))

    # Load the data into a clean text file, one line per entry.
    output_path = "rick_corpus.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_rick_lines:
            f.write(line + "\n")
    
    print(f"\n✅ Success! Scraped {len(all_rick_lines)} lines.")
    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()