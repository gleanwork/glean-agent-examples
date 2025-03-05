from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
import requests
from pydantic import Field

class GleanRetriever(BaseRetriever):
    """Retriever that uses Glean's search API."""
    
    glean_api_url: str = Field(description="URL for the Glean API")
    api_key: str = Field(description="API key for Glean")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    
    def __init__(
        self, 
        glean_api_url: str,
        api_key: str,
        max_results: int = 5
    ):
        """Initialize the GleanRetriever."""
        super().__init__(
            glean_api_url=glean_api_url,
            api_key=api_key,
            max_results=max_results
        )
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query."""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            payload = {
                "query": query,
                "limit": self.max_results,
                "filters": {}  # Add custom filters if needed
            }
            
            response = requests.post(
                f"{self.glean_api_url}/search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            search_results = response.json()
            
            # Convert Glean results to LangChain Documents
            documents = []
            for result in search_results.get("results", []):
                # Extract the content and metadata from Glean's response
                content = result.get("snippet", "")
                if not content and "content" in result:
                    content = result.get("content", "")
                
                # Create metadata dictionary from Glean result properties
                metadata = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "source": "glean",
                    "score": result.get("score", 0.0),
                    "document_type": result.get("type", "unknown")
                }
                
                # Additional metadata that might be available
                if "lastModified" in result:
                    metadata["last_modified"] = result["lastModified"]
                if "author" in result:
                    metadata["author"] = result["author"]
                
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))
            
            return documents
        
        except Exception as e:
            print(f"Error retrieving documents from Glean: {str(e)}")
            return []