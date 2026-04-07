import json
import os
import structlog
from typing import List, Dict, Any

logger = structlog.get_logger(__name__)

class PreferenceManager:
    """
    Manages ranked preference pairs, performing bias checks and duplicate detection.
    """
    def __init__(self):
        pass

    def process_ranking(
        self, 
        raw_ranked_data: List[Dict[str, Any]], 
        output_path: str = "data/rlhf/preferences.jsonl"
    ):
        """
        Expects a list where each item has:
        {
           "query": "...",
           "chosen": "...",
           "rejected": "..."
        }
        """
        seen_pairs = set()
        final_pairs = []
        bias_alerts = 0

        for item in raw_ranked_data:
            query = item["query"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            # 1. Duplicate Detection
            pair_hash = hash((query, chosen, rejected))
            if pair_hash in seen_pairs:
                logger.warning("duplicate_pair_skipped", query=query[:30])
                continue
            seen_pairs.add(pair_hash)

            # 2. Length Bias Check
            len_ratio = len(chosen) / (len(rejected) + 1e-6)
            if len_ratio > 3.0:
                logger.info("length_bias_detected", ratio=len_ratio, query=query[:30])
                bias_alerts += 1
            
            final_pairs.append({
                "query": query,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "len_ratio": round(len_ratio, 2)
                }
            })

        # Save as JSONL
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            for pair in final_pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info("preferences_imported", 
                    count=len(final_pairs), 
                    duplicates_removed=len(raw_ranked_data) - len(final_pairs),
                    length_bias_alerts=bias_alerts)
        
        return len(final_pairs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked_json", required=True, help="Path to ranked results from human")
    parser.add_argument("--output", default="data/rlhf/preferences.jsonl")
    args = parser.parse_args()

    with open(args.ranked_json, "r") as f:
        ranked_data = json.load(f)

    manager = PreferenceManager()
    manager.process_ranking(ranked_data, output_path=args.output)
