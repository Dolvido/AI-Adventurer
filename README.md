# AI Adventurer

The `AIAdventurer` class is designed to create an intelligent agent that can navigate and interact with its environment, make decisions, and take actions. It utilizes the BART (Bidirectional and Auto-Regressive Transformers) model from the Hugging Face Transformers library to generate thoughts and actions based on the current situation.

## Features

- Initializes an AI adventurer with a name, character class, race, and background
- Loads a character-specific knowledge base
- Maintains a memory of past thoughts and actions
- Perceives the current situation
- Generates thoughts based on the knowledge base and current situation
- Generates actions based on the generated thoughts

## Usage

```python
from AIAdventurer import AIAdventurer

# Create an instance of the AIAdventurer
mage = AIAdventurer("Gandalf", "Mage", "Human", "Sage")

# Example usage
situation = "You find yourself in a dark dungeon. There is a locked door ahead of you."
mage.perceive(situation)
thought = mage.think()
print(f"Thought: {thought}")
action = mage.act(thought)
print(f"Action: {action}")
```

## Dependencies

- `torch`
- `transformers` (specifically `BartTokenizer` and `BartForConditionalGeneration`)

## Future Improvements

- Implement a more comprehensive knowledge base and decision-making logic
- Integrate the agent with a game engine or interactive environment
- Enhance the memory and learning capabilities of the agent
- Explore multi-agent interactions and team-based dynamics

## Contributions

Contributions to this project are welcome. Feel free to submit issues, feature requests, or pull requests on the [GitHub repository](https://github.com/your-username/AI-Adventurer).