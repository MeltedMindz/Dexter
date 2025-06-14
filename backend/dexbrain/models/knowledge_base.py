import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import asyncio
import aiofiles
from ..config import Config


class KnowledgeBase:
    """Manages storage and retrieval of AI agent insights"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Config.KNOWLEDGE_DB_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    async def store_insight(self, category: str, insight: Dict[str, Any]) -> None:
        """Store an insight with timestamp
        
        Args:
            category: Category/type of insight
            insight: Data to store
        """
        insight['timestamp'] = datetime.now().isoformat()
        insight['id'] = f"{category}_{datetime.now().timestamp()}"
        
        # Create category file if not exists
        file_path = self.storage_path / f"{category}.jsonl"
        
        async with self._lock:
            async with aiofiles.open(file_path, 'a') as f:
                await f.write(json.dumps(insight) + '\n')
    
    async def retrieve_insights(
        self, 
        category: str, 
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent insights for a category
        
        Args:
            category: Category to retrieve
            limit: Maximum number of insights to return
            start_date: Filter insights after this date
            end_date: Filter insights before this date
            
        Returns:
            List of insights matching criteria
        """
        file_path = self.storage_path / f"{category}.jsonl"
        
        if not file_path.exists():
            return []
        
        insights = []
        async with aiofiles.open(file_path, 'r') as f:
            async for line in f:
                try:
                    insight = json.loads(line.strip())
                    
                    # Apply date filters if provided
                    if start_date or end_date:
                        insight_date = datetime.fromisoformat(insight['timestamp'])
                        if start_date and insight_date < start_date:
                            continue
                        if end_date and insight_date > end_date:
                            continue
                    
                    insights.append(insight)
                except json.JSONDecodeError:
                    continue
        
        # Return last 'limit' insights
        return insights[-limit:]
    
    async def get_categories(self) -> List[str]:
        """Get all available insight categories
        
        Returns:
            List of category names
        """
        categories = []
        for file_path in self.storage_path.glob("*.jsonl"):
            categories.append(file_path.stem)
        return categories
    
    async def clear_category(self, category: str) -> None:
        """Clear all insights for a category
        
        Args:
            category: Category to clear
        """
        file_path = self.storage_path / f"{category}.jsonl"
        if file_path.exists():
            file_path.unlink()
    
    async def get_insight_count(self, category: str) -> int:
        """Get number of insights in a category
        
        Args:
            category: Category to count
            
        Returns:
            Number of insights
        """
        file_path = self.storage_path / f"{category}.jsonl"
        if not file_path.exists():
            return 0
        
        count = 0
        async with aiofiles.open(file_path, 'r') as f:
            async for _ in f:
                count += 1
        
        return count