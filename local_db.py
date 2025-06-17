import json
import os
from typing import List, Dict, Any
import pandas as pd

class LocalDatabase:
    """Simple file-based database adapter to replace Neo4j temporarily."""
    
    def __init__(self, db_path: str = "local_database.json"):
        self.db_path = db_path
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading database: {e}")
                return {"records": [], "tables": []}
        return {"records": [], "tables": []}
    
    def _save_data(self):
        """Save data to JSON file."""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error saving database: {e}")
    
    def clear_all(self):
        """Clear all data."""
        self.data = {"records": [], "tables": []}
        self._save_data()
        return True
    
    def load_csv(self, csv_path: str):
        """Load CSV data into local database."""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            table_name = os.path.basename(csv_path)
            
            # Convert DataFrame to records
            records = []
            for idx, row in df.iterrows():
                record = {"id": str(idx)}
                for col, value in row.items():
                    # Handle NaN values
                    if pd.isna(value):
                        record[col] = None
                    else:
                        record[col] = str(value)
                records.append(record)
            
            # Store in local database
            self.data["records"] = records
            if table_name not in self.data["tables"]:
                self.data["tables"].append(table_name)
            
            self._save_data()
            print(f"✅ Loaded {len(records)} records from {table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def get_record_count(self) -> int:
        """Get total number of records."""
        return len(self.data.get("records", []))
    
    def search_records(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Simple text search across all records."""
        query_lower = query.lower()
        results = []
        
        for record in self.data.get("records", []):
            # Search across all text fields
            match_score = 0
            for key, value in record.items():
                if value and isinstance(value, str):
                    if query_lower in value.lower():
                        match_score += 1
            
            if match_score > 0:
                results.append({
                    "node": record,
                    "score": float(match_score)
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_all_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all records with optional limit."""
        records = self.data.get("records", [])[:limit]
        return [{"node": record, "score": 1.0} for record in records]
    
    def get_sample_records(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample records for schema detection."""
        records = self.data.get("records", [])[:limit]
        return [{"node": record} for record in records]
    
    def filter_by_field(self, field: str, value: str = None, sort_direction: str = "ASC", limit: int = 20) -> List[Dict[str, Any]]:
        """Filter and sort records by a specific field."""
        results = []
        
        for record in self.data.get("records", []):
            if field in record and record[field] is not None and str(record[field]).strip():
                field_value = record[field]
                
                # If searching for specific value
                if value and value.lower() not in str(field_value).lower():
                    continue
                
                # Try to convert to numeric for sorting
                try:
                    sort_key = float(field_value)
                except (ValueError, TypeError):
                    sort_key = str(field_value).lower()
                
                results.append({
                    "node": record,
                    "score": 1.0,
                    "sort_key": sort_key
                })
        
        # Sort by the field
        reverse_sort = sort_direction.upper() == "DESC"
        results.sort(key=lambda x: x["sort_key"], reverse=reverse_sort)
        
        # Remove sort_key before returning
        for result in results:
            del result["sort_key"]
        
        return results[:limit]
    
    def count_by_field(self, field: str, sort_direction: str = "DESC", limit: int = 20) -> List[Dict[str, Any]]:
        """Count occurrences by field value."""
        counts = {}
        
        for record in self.data.get("records", []):
            if field in record and record[field] is not None:
                value = str(record[field]).strip()
                if value and value != "null" and value != "NA":
                    if value not in counts:
                        counts[value] = {"count": 0, "sample": record}
                    counts[value]["count"] += 1
        
        # Convert to results format
        results = []
        for value, data in counts.items():
            results.append({
                "node": {
                    "brand": value,
                    "count": data["count"],
                    "sample": data["sample"]
                },
                "score": float(data["count"])
            })
        
        # Sort by count
        reverse_sort = sort_direction.upper() == "DESC"
        results.sort(key=lambda x: x["score"], reverse=reverse_sort)
        
        return results[:limit]

# Global instance
local_db = LocalDatabase() 