import re
from typing import List, Dict, Any

class HeadingAwareChunker:
    """
    Chunks text by looking for headings (markdown style #, ##, etc.)
    Ensures that headers are preserved or prepended to chunks for context.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """
        Naive implementation of heading-aware chunking.
        Splits by headers first, then sub-chunks if needed.
        """
        # Split by markdown headers
        sections = re.split(r"(^#+\s+.*)", text, flags=re.MULTILINE)
        
        chunks = []
        current_header = ""
        
        for section in sections:
            if not section.strip():
                continue
                
            if section.startswith("#"):
                current_header = section.strip()
                continue
            
            # Process the content under the header
            content = f"{current_header}\n{section}" if current_header else section
            
            if len(content) <= self.chunk_size:
                chunks.append(content)
            else:
                # Simple recursive-like split for large sections
                start = 0
                while start < len(content):
                    end = start + self.chunk_size
                    chunk_text = content[start:end]
                    # Try to break at newline if not at the very end
                    if end < len(content):
                        last_nl = chunk_text.rfind('\n')
                        if last_nl > self.chunk_size // 2:
                            end = start + last_nl
                            chunk_text = content[start:end]
                    
                    chunks.append(chunk_text)
                    start += (self.chunk_size - self.chunk_overlap)
                    
        return chunks
