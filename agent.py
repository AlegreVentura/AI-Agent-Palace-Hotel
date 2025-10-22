
from pathlib import Path
import getpass
import os
from typing import Annotated, Optional, TypedDict, Literal
from langgraph.graph.message import AnyMessage, add_messages
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# Use Google Gemini
from tools import (
    consult_policy,
    frequent_questions,
    db_get,
    check_server_health,
    get_stats,
    calculate_price,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from pathlib import Path
from typing import Optional

# Load environment variables from .env file
load_dotenv()

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


_set_env("GOOGLE_API_KEY")


# Load system prompt
SYSTEM_PROMPT_PATH = Path(__file__).parent / "config" / "system_prompt.txt"

def load_system_prompt() -> str:
    """Load the system prompt from config/system_prompt.txt"""
    try:
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: System prompt file not found at {SYSTEM_PROMPT_PATH}")
        return "You are a helpful AI assistant for Palace Resorts call center."


SYSTEM_PROMPT = load_system_prompt()

# Using Google Gemini 2.5 Pro
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7
)



def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str
    tail_text: str  # last ~60s of transcript
    slots: dict     # {"dates": "...", "guests": 2, ...}
    stage: str      # "discovery|qualification|offer|closing"
    json_output: Optional[dict]  # JSON from main agent (without summary)
    final_output: Optional[dict]  # Complete JSON with summary from summary agent


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            customer_info = configuration.get("customer_info", None)
            agent_info = configuration.get("agent_info", None)
            
            # Build user context
            user_context = self._build_user_context(customer_info, agent_info)
            state = {**state, "user_info": user_context}
            
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            tool_calls = getattr(result, 'tool_calls', [])
            if not tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        
        # Try to extract JSON from the response
        json_output = self._extract_json_from_response(result)
        
        return {"messages": result, "json_output": json_output}
    
    def _extract_json_from_response(self, result) -> Optional[dict]:
        """Extract JSON from the LLM response"""
        import json
        import re
        
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Try to find JSON in the response
        try:
            # First try to parse the entire content as JSON
            json_data = json.loads(content)
            # Ensure resumen_llamada_md is empty string
            if "resumen_llamada_md" not in json_data:
                json_data["resumen_llamada_md"] = ""
            if not json_data.get("resumen_llamada_md"):
                json_data["resumen_llamada_md"] = ""

            
            # Ensure emocion_principal is -1 (will be set by summary agent)
            if "analisis_emociones" not in json_data:
                json_data["analisis_emociones"] = {"emocion_principal": -1}
            elif "emocion_principal" not in json_data["analisis_emociones"]:
                json_data["analisis_emociones"]["emocion_principal"] = -1
            elif json_data["analisis_emociones"]["emocion_principal"] != -1:
                json_data["analisis_emociones"]["emocion_principal"] = -1
            
            return json_data
        except:
            # Try to find JSON block in the content
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group(0))
                    # Ensure resumen_llamada_md is empty string
                    if "resumen_llamada_md" not in json_data:
                        json_data["resumen_llamada_md"] = ""
                    if not json_data.get("resumen_llamada_md"):
                        json_data["resumen_llamada_md"] = ""

                    
                    # Ensure emocion_principal is -1 (will be set by summary agent)
                    if "analisis_emociones" not in json_data:
                        json_data["analisis_emociones"] = {"emocion_principal": -1}
                    elif "emocion_principal" not in json_data["analisis_emociones"]:
                        json_data["analisis_emociones"]["emocion_principal"] = -1
                    elif json_data["analisis_emociones"]["emocion_principal"] != -1:
                        json_data["analisis_emociones"]["emocion_principal"] = -1
                    
                    return json_data
                except:
                    pass
        
        return None
    
    def _build_user_context(self, customer_info: dict, agent_info: dict) -> str:
        """Build formatted user context from customer and agent information"""
        context_parts = []
        
        if customer_info:
            context_parts.append("CUSTOMER INFORMATION:")
            for key, value in customer_info.items():
                if value:
                    context_parts.append(f"  {key}: {value}")
        
        if agent_info:
            context_parts.append("\nAGENT INFORMATION:")
            for key, value in agent_info.items():
                if value:
                    context_parts.append(f"  {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else "No user information available"



# Build the primary assistant prompt with loaded system prompt and examples
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}"
            "\n\n--- CURRENT CONTEXT ---"
            "\nCurrent user:\n<User>\n{user_info}\n</User>"
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(
    system_prompt=SYSTEM_PROMPT,
    time=datetime.now
)

# Define tools for Palace Resorts operations
part_1_tools = [
    # Policy/FAQ helpers
    consult_policy,
    frequent_questions,
    # Backend API wrappers
    db_get,
    check_server_health,
    get_stats,
    calculate_price
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Create a separate prompt and LLM for the summary agent
summary_agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a specialized summary and emotion analyzer for Palace Resorts call center.
Your job is to:
1. Generate the 'resumen_llamada_md' field based on the JSON data provided
2. Analyze and assign the 'emocion_principal' value based on the conversation tone

You will receive a JSON object with all the call information. 

**EMOTION ANALYSIS:**
Analyze the overall emotional tone and assign the appropriate value:
- 0 - Neutral (informational, no clear emotional tone)
- 1 - Excited (enthusiastic, eager to book, positive energy)
- 2 - Anxious (nervous, has concerns or questions)
- 3 - Frustrated (experiencing issues, showing impatience)
- 4 - Disappointed (objections present, needs problem solving)
- 5 - Satisfied (positive interaction, follow-up opportunity)

**SUMMARY FORMAT:**
Generate a concise, well-formatted summary in **English** using this EXACT structure:

```markdown
### Call Summary
**Client:** [Full Name] ([customer type], [reservation type])
**Resort:** **[Resort Name]** ([City], [Country])
**Dates:** **[Check-in date–Check-out date]** · **Type:** *[room_type]* · **Availability:** **Yes/No** ( **[N]** remaining)
**Avg. price/night:** **USD [price]** · **Estimated total:** **USD [total]** · **Loyalty savings:** **USD [savings]**
**Systems:** status **[healthy/degraded]**, version **[version]**

> **Next step:** [actionable next step for agent]
```

Keep it scannable, use bold for key info, and always end with a concrete next step.

**IMPORTANT:** Return a JSON object with TWO fields (use double curly braces for JSON structure):
{{
  "emocion_principal": [0-5],
  "resumen_llamada_md": "[markdown summary]"
}}"""
        ),
        ("placeholder", "{messages}"),
    ]
)

summary_agent_runnable = summary_agent_prompt | llm


class SummaryAgent:
    """Agent responsible for generating the call summary and emotion analysis based on collected information"""
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        """Generate summary and emotion analysis from the JSON output of the main agent"""
        json_output = state.get("json_output", {})
        
        if not json_output:
            # If no JSON output yet, skip summary generation
            return {"final_output": None}
        
        # Create a prompt for the summary agent with the JSON data
        json_str = str(json_output)
        summary_prompt = f"""Analyze the following call data and provide:
1. The emotional tone (emocion_principal: 0-5)
2. A markdown summary (resumen_llamada_md)

Call data:
{json_str}

Return a JSON object with both fields."""
        
        # Get the summary and emotion from the LLM
        result = self.runnable.invoke({"messages": [("user", summary_prompt)]})
        
        # Extract the response
        import json
        import re
        
        content = result.content if hasattr(result, 'content') else str(result)
        
        # Try to parse the JSON response from the summary agent
        try:
            # Try to parse as JSON first
            summary_data = json.loads(content)
            emotion = summary_data.get("emocion_principal", 0)
            summary_text = summary_data.get("resumen_llamada_md", "")
        except:
            # If not JSON, try to extract both fields
            emotion = 0
            summary_text = content
            
            # Try to find emotion value
            emotion_match = re.search(r'"emocion_principal"\s*:\s*(\d+)', content)
            if emotion_match:
                emotion = int(emotion_match.group(1))
            
            # Try to find summary text
            summary_match = re.search(r'"resumen_llamada_md"\s*:\s*"([^"]+)"', content, re.DOTALL)
            if summary_match:
                summary_text = summary_match.group(1)
            else:
                # If can't find in JSON format, use the whole content as summary
                summary_text = content
        
        # Update the JSON output with both emotion and summary
        final_output = {**json_output}
        final_output["analisis_emociones"]["emocion_principal"] = emotion
        final_output["resumen_llamada_md"] = summary_text
        
        return {"final_output": final_output, "messages": [result]}


########### LangGraph components
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

def route_after_assistant(state: State) -> str:
    """Decide where to go after assistant responds"""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        tool_calls = getattr(last_message, 'tool_calls', [])
        if tool_calls:
            return "tools"
    
    # If we have JSON output, move to summary generation
    if state.get("json_output"):
        return "summary_agent"
    
    # Otherwise, end (shouldn't normally reach here)
    return END

builder = StateGraph(State)

# nodes
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", ToolNode(part_1_tools))
builder.add_node("summary_agent", SummaryAgent(summary_agent_runnable))

# edges: these determine how the control flow moves
builder.add_edge(START, "assistant")

# Conditional edge from assistant: either go to tools or summary
builder.add_conditional_edges(
    "assistant",
    route_after_assistant,
    {
        "tools": "tools",
        "summary_agent": "summary_agent",
        END: END
    }
)

builder.add_edge("tools", "assistant")
builder.add_edge("summary_agent", END)

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = InMemorySaver()
part_1_graph = builder.compile(checkpointer=memory)

def run_agent(call_transcription: str, customer_info: dict = None, agent_info: dict = None):
    config = {
        "configurable": {
            "customer_info": customer_info or {},
            "agent_info": agent_info or {},
            "thread_id": "1",
        }
    }

    input_message = {
        "messages": [("user", call_transcription)]
    }

    print("\n🧠 Running LangGraph pipeline...")
    result = part_1_graph.invoke(input_message, config)

    # Save partial or final results
    output = {}
    if result.get("final_output"):
        output = result["final_output"]
    elif result.get("json_output"):
        output = result["json_output"]
    else:
        output = result

    save_json_state(output)

    return output

    
import json, os

def save_json_state(data: dict, filename: str = "current_call_state.json"):
    """Write the current agent state to a JSON file (overwrites each time)."""
    os.makedirs("call_state", exist_ok=True)
    path = os.path.join("call_state", filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Error writing {path}: {e}")


if __name__ == "__main__":
    # Test the agent with example data
    test_customer_info = {
        "name": "Maria Rodriguez",
        "age": 35,
        "party_size": "2 adults",
        "travel_purpose": "10th Anniversary + Birthday celebration",
        "check_in": "October 21, 2025",
        "check_out": "October 25, 2025",
        "nights": 4,
        "room_preference": "Ocean view",
        "dietary_restrictions": "Vegetarian (both guests)"
    }
    
    test_agent_info = {
        "name": "Sarah Johnson",
        "id": "AGT-001",
        "location": "Cancun Call Center"
    }
    
    test_call = """
    Hi, yes, I'm Maria Rodriguez. I'm looking to book a room for my husband and me. 
    We're celebrating our 10th wedding anniversary next month on October 21st through the 25th.
    I'm turning 35 that week too, so it's a double celebration! 
    We're looking for something romantic with an ocean view if possible. We're vegetarians, by the way.
    """
    
    print("Testing AI Copilot Agent...")
    print("=" * 80)
    response = run_agent(test_call, test_customer_info, test_agent_info)
    print("\nAgent Response:")
    
    # Pretty print the JSON response
    import json
    if isinstance(response, dict):
        print(json.dumps(response, indent=2, ensure_ascii=False))
    else:
        print(response)