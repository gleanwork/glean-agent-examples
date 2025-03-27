import json
import sys
from glean_agent_examples.common import BaseExampleRunner, IconType
from glean_agent_examples.examples.langgraph.glean_chat_model_server import LangGraphGleanChatExample


class LangGraphGleanChatRunner(BaseExampleRunner):
    """
    Runner for the LangGraph Glean Chat Model example.
    
    This runner implements the validation and testing methods specific to the
    Glean Chat Model example.
    """
    
    def _validate_glean_connection(self, client) -> bool:
        """
        Validate connection to Glean Chat API using the provided client.
        
        Args:
            client: Authenticated GleanClient instance
            
        Returns:
            True if validation succeeds, False otherwise
        """
        
        payload = {
            "messages": [
                {
                    "author": "USER",
                    "messageType": "CONTENT",
                    "agentConfig": {
                        "agent": "DEFAULT",
                        "mode": "DEFAULT"
                    },
                    "fragments": [
                        {
                            "text": "Hello, can you tell me about the company holidays?"
                        }
                    ]
                }
            ]
        }
        
        self.print_title("Making chat request", IconType.API)
        
        response = client.post("chat", json=payload)
        
        if "messages" in response and len(response.get("messages", [])) > 0:
            last_message = response["messages"][-1]
            if "fragments" in last_message and len(last_message["fragments"]) > 0:
                self.print_title("Glean Chat API connection successful!", IconType.SUCCESS)

                return True
            else:
                self.print_message("Glean Chat API connected but no message fragments were returned.", IconType.WARNING)
                self.print_message(f"Response: {json.dumps(response, indent=2)}")
                
                return False
        else:
            self.print_message("Glean Chat API connected but no messages were returned.", IconType.WARNING)
            self.print_message(f"Response: {json.dumps(response, indent=2)}")
            return False

if __name__ == "__main__":
    example = LangGraphGleanChatExample()
    runner = LangGraphGleanChatRunner(example)
    
    try:
        query = runner.read_query()
        runner.run(query)
    except ValueError as e:
        runner.print_message(f"Error: {e}")
        runner.print_message('Usage: task run:example EXAMPLE=langgraph/glean_chat_model "What is your question?"')
        sys.exit(1)
