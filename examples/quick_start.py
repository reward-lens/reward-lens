"""
Example: Quick Reward Lens in 5 Lines

The simplest possible reward-lens workflow — load a model, analyze a
preference pair, get a plot. Five lines of code.

Usage:
    python examples/quick_start.py
"""

from reward_lens import RewardModel, reward_lens_plot

# Load a reward model (this downloads from HuggingFace on first run)
model = RewardModel.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")

# Define a preference pair
prompt = "Explain what a neural network is."
good = (
    "A neural network is a computing system inspired by biological neural networks "
    "in the brain. It consists of layers of interconnected nodes (neurons) that "
    "process information. Data flows through the network, with each connection "
    "having a learned weight that determines its influence. Neural networks learn "
    "by adjusting these weights based on training data, allowing them to recognize "
    "patterns and make predictions."
)
bad = (
    "Neural networks are basically just math and computers doing stuff together. "
    "They're used in AI. It's complicated technology."
)

# Run reward lens analysis and plot — this shows WHERE in the model
# the preference between the good and bad responses forms.
result = reward_lens_plot(model, prompt, good, bad, save_path="reward_lens_quickstart.png")

print(f"\nPreferred score:    {result.reward_preferred:.4f}")
print(f"Dispreferred score: {result.reward_dispreferred:.4f}")
print(f"Preference Δ:       {result.reward_preferred - result.reward_dispreferred:+.4f}")
print(f"Crystallization:    Layer {result.crystallization_layer}")
