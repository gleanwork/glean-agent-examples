import sys
import json
from glean_agent_examples.common import BaseExampleRunner, IconType
from glean_agent_examples.examples.langgraph.glean_search_retriever_server import LangGraphGleanSearchExample


class LangGraphGleanSearchRunner(BaseExampleRunner):
    """
    Runner for the LangGraph Glean Search Retriever example.
    
    This runner implements the validation and testing methods specific to the
    Glean Search Retriever example.
    """
    
    def _validate_glean_connection(self, client) -> bool:
        """
        Validate connection to Glean Search API using the provided client.
        
        Args:
            client: Authenticated GleanClient instance
            
        Returns:
            True if validation succeeds, False otherwise
        """
        
        payload = {
            "query": "test",
            "pageSize": 1
        }
        
        self.print_title("Making search request", IconType.API)
        results = client.post("search", json=payload)
        
        if "results" in results:
            result_count = len(results.get("results", []))
            self.print_message(f"Glean client connection successful! Got {result_count} result(s).", IconType.SUCCESS)
            return True
        else:
            self.print_message("Glean client connected but no results were returned.", IconType.WARNING)
            self.print_message(f"Response: {json.dumps(results, indent=2)}")
            return True  # Still consider this a success since we connected

if __name__ == "__main__":
    example = LangGraphGleanSearchExample()
    runner = LangGraphGleanSearchRunner(example)
    
    try:
        query = runner.read_query()
        runner.run(query)
    except ValueError as e:
        runner.print_message(f"Error: {e}")
        runner.print_message('Usage: task run:example EXAMPLE=langgraph/glean_search_retriever "What is your question?"')
        sys.exit(1)
