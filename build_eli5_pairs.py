import asyncio
import json
import os
import time
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_batch(num_to_generate, batch_number):
    """Generates a small batch of prompts to avoid token limits."""
    print(f"Generating Batch {batch_number} ({num_to_generate} topics)...")
    
    system_instruction = (
        "You are a research assistant building a dataset for mechanistic interpretability. "
        "Generate a list of unique, diverse topics (70% technical, 30% general). "
        "For each topic, provide a 'Novice' framing and an 'Expert' framing."
    )
    
    user_request = (
        f"Generate {num_to_generate} UNIQUE topics that are different from common ones. "
        "For each, output a JSON object with: 'topic', 'domain', 'novice_prompt', and 'expert_prompt'.\n"
        "Return the output as a JSON object with a key 'topics' containing the list."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_request}
            ],
            response_format={ "type": "json_object" }
        )
        
        batch_data = json.loads(response.choices[0].message.content)
        return batch_data.get("topics", [])

    except Exception as e:
        print(f"Error in Batch {batch_number}: {e}")
        return []

if __name__ == "__main__":
    TOTAL_TOPICS = 250
    BATCH_SIZE = 50
    all_prompts = []

    for i in range(0, TOTAL_TOPICS, BATCH_SIZE):
        batch_num = (i // BATCH_SIZE) + 1
        batch_results = generate_batch(BATCH_SIZE, batch_num)
        all_prompts.extend(batch_results)
        
        # Optional: slight delay to ensure you don't hit rate limits
        time.sleep(1) 

    # Save final results
    with open("generated_prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=4)
        
    print(f"\nSuccess! Total prompt pairs generated: {len(all_prompts)}")
    print("Saved to 'generated_prompts.json'")