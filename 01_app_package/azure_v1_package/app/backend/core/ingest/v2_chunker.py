import re
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from core.ingest.chunker import HeadingAwareChunker

class V2Chunker(HeadingAwareChunker):
    """
    Experimental Chunker for Version 2.
    Adds support for converting HTML and CSV tables into Markdown format
    to preserve structural data during ingestion.
    """
    
    def html_tables_to_markdown(self, html: str) -> str:
        """
        Extracts HTML tables and converts them to Markdown.
        """
        soup = BeautifulSoup(html, "html.parser")
        tables = soup.find_all("table")
        
        for table in tables:
            try:
                from io import StringIO
                # Use pandas to convert HTML table to a DataFrame
                df = pd.read_html(StringIO(str(table)))[0]
                # Convert DataFrame to Markdown
                markdown_table = "\n" + df.to_markdown(index=False) + "\n"
                # Replace the original table with markdown in the soup
                table.replace_with(markdown_table)
            except Exception:
                # If conversion fails, keep original or strip tags
                continue
                
        return str(soup)

    def csv_content_to_markdown(self, text: str) -> str:
        """
        Detects if content is CSV-like and converts to Markdown if so.
        Simple heuristic: check for multiple commas in multiple lines.
        """
        lines = text.strip().split("\n")
        if len(lines) > 2 and all(line.count(",") >= 2 for line in lines[:3]):
            try:
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                return "\n" + df.to_markdown(index=False) + "\n"
            except Exception:
                return text
        return text

    def chunk(self, text: str) -> List[str]:
        """
        Enriched chunking with structural data preservation.
        """
        # 1. Pre-process structural data
        processed_text = text
        if "</table>" in text.lower():
            processed_text = self.html_tables_to_markdown(processed_text)
            
        # 2. Use base class chunking logic on the enriched text
        return super().chunk(processed_text)
