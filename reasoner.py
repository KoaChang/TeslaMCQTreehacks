import os
import csv
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from asyncio import Semaphore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncMultipleChoiceReasoner:
    def __init__(self, api_key: str, max_concurrent_requests: int = 5):
        """
        Initialize the multiple-choice reasoner with API credentials
        
        Args:
            api_key (str): OpenAI API key
            max_concurrent_requests (int): Maximum number of concurrent API requests
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = Semaphore(max_concurrent_requests)
    
    async def reason_over_description(self, description: str, question: str, video_id: str) -> Dict:
        """
        Sends description and question to the 'o1-preview' model to perform reasoning
        and choose the best multiple-choice answer.
        
        Args:
            description (str): Text describing video frames
            question (str): The multiple-choice question that needs to be answered
            video_id (str): ID of the video for logging purposes
            
        Returns:
            Dict: API response with the chosen answer
        """
        async with self.semaphore:  # Limit concurrent requests
            content_prompt = f"""Here is the multiple choice question that was posed:
{question}

Here is the vision model's complete analysis:
{description}

Now, review the analysis provided by the vision model above, which includes:
1. A frame-by-frame analysis of 5 dashcam video frames
2. Contextual reasoning about the scene
3. Analysis of multiple choice options
4. Their final answer selection

Your task is to:
1. Carefully read through the vision model's observations and reasoning
2. Consider whether their frame-by-frame analysis captures all relevant details
3. Evaluate if their contextual reasoning is sound and complete
4. Assess if their analysis of the multiple choice options is thorough and logical
5. Determine if their final answer choice is the most appropriate given the evidence

Then:
1. If you agree with the vision model's answer, output their answer choice in <answer></answer> tags
2. If you disagree with their answer, output your chosen answer in <answer></answer> tags, followed by a brief explanation of why you chose differently

Note: Focus on concrete details mentioned in the vision model's analysis rather than making assumptions about what might be in the frames but wasn't mentioned.
"""
            
            try:
                completion = await self.client.chat.completions.create(
                    model="o1-preview",
                    messages=[
                        {
                            "role": "user",
                            "content": content_prompt
                        }
                    ]
                )
                
                return {
                    "video_id": video_id,
                    "answer": completion.choices[0].message.content,
                    "finish_reason": completion.choices[0].finish_reason
                }
            except Exception as e:
                logger.error(f"Error processing video ID {video_id}: {str(e)}")
                return {
                    "video_id": video_id,
                    "error": str(e)
                }

    async def process_batch(self, items: List[Dict]) -> List[Dict]:
        """
        Process a batch of descriptions and questions concurrently
        
        Args:
            items (List[Dict]): List of dictionaries containing video_id, description, and question
            
        Returns:
            List[Dict]: List of results
        """
        tasks = []
        for item in items:
            task = self.reason_over_description(
                item['description'], 
                item['question'],
                item['video_id']
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

async def main():
    load_dotenv()  # Load environment variables from .env if available
    api_key = os.getenv('OPENAI_API_KEY_KOA_O1')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_KOA_O1 environment variable not set")
    
    # Initialize reasoner with concurrent request limit
    reasoner = AsyncMultipleChoiceReasoner(api_key=api_key, max_concurrent_requests=5)
    
    # Folders for inputs/outputs
    description_dir = Path('initial_answers')
    final_dir = Path('final_answers')
    final_dir.mkdir(exist_ok=True)
    
    # Prepare batch of items to process
    items_to_process = []
    
    # Read the CSV and gather all items
    with open('questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            video_id = row['id']
            question = row['question']
            
            # Load the description JSON
            description_file = description_dir / f"{video_id}_result.json"
            
            if not description_file.is_file():
                logger.error(f"No description JSON found for video ID {video_id} at {description_file}")
                continue
            
            try:
                with open(description_file, 'r') as f:
                    description_data = json.load(f)
                
                description_text = description_data.get("answer", "")
                
                if not description_text:
                    logger.error(f"No description text found in {description_file}")
                    continue
                
                items_to_process.append({
                    'video_id': video_id,
                    'description': description_text,
                    'question': question
                })
                
            except Exception as e:
                logger.error(f"Error reading description file for video ID {video_id}: {str(e)}")
                continue
    
    # Process all items in parallel
    logger.info(f"Processing {len(items_to_process)} items in parallel...")
    results = await reasoner.process_batch(items_to_process)
    
    # Save results
    for result in results:
        video_id = result['video_id']
        output_path = final_dir / f"{video_id}_result.json"
        
        # Remove video_id from the result before saving
        result_to_save = {k: v for k, v in result.items() if k != 'video_id'}
        
        with open(output_path, 'w') as outf:
            json.dump(result_to_save, outf, indent=2)
        
        logger.info(f"Completed reasoning for video ID: {video_id}")

if __name__ == "__main__":
    asyncio.run(main())