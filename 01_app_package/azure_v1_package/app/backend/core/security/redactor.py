import re
from typing import List, Dict, Any, Tuple

class Redactor:
    """
    Handles PII and secret redaction using regex patterns.
    Can be configured per tenant (scaffolded here with defaults).
    """
    DEFAULT_PATTERNS = {
        "employee_id": r"\b\d{8,}\b", # 8 or more digits
        "customer_name": r"(?i)Customer:\s*([A-Z0-9\s]+)",
        "api_key": r"(?i)api[-_]?key[:\s]+([a-zA-Z0-9]{32,})",
    }

    def __init__(self, tenant_patterns: Dict[str, str] = None):
        self.patterns = self.DEFAULT_PATTERNS.copy()
        if tenant_patterns:
            self.patterns.update(tenant_patterns)

    def redact(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Redacts sensitive patterns from text.
        Returns (redacted_text, redaction_report).
        """
        report = {}
        redacted_text = text
        
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                report[name] = len(matches)
                # If pattern has a capturing group, we only redact the group (for Customer: XYZ)
                if "(" in pattern and ")" in pattern:
                    # Specialized redaction for Customer: XYZ -> Customer: [REDACTED]
                    def substitute(match):
                        # Construct a replacement string preserving the non-captured part
                        # This is a bit simplified; for production we'd use more robust logic
                        full_match = match.group(0)
                        captured = match.group(1)
                        return full_match.replace(captured, "[REDACTED]")
                    
                    redacted_text = re.sub(pattern, substitute, redacted_text)
                else:
                    redacted_text = re.sub(pattern, "[REDACTED]", redacted_text)
                    
        return redacted_text, report
