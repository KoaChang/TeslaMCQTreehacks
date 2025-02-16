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
    
    async def reason_over_attempts(self, attempts: List[Dict], question: str, video_id: str) -> Dict:
        """
        Sends multiple reasoning attempts and question to the 'o1-preview' model to evaluate
        and choose the best answer among them.
        
        Args:
            attempts (List[Dict]): List of previous reasoning attempts
            question (str): The multiple-choice question that needs to be answered
            video_id (str): ID of the video for logging purposes
            
        Returns:
            Dict: API response with the chosen answer and explanation
        """
        async with self.semaphore:  # Limit concurrent requests
            # Format attempts for the prompt
            formatted_attempts = ""
            for attempt in attempts:
                formatted_attempts += f"\nAttempt {attempt['attempt_number']}:\n{attempt['answer']}\n"

            content_prompt = f"""Here is the multiple choice question that was posed:
{question}

I have received three different reasoning attempts from a vision model analyzing the same video frames:
{formatted_attempts}
            
Your task is to:
1. Carefully review all three reasoning attempts
2. Aggregate and compare the frame-by-frame observations across all attempts
3. Consider any unique details or perspectives provided by each attempt
4. Evaluate the strength and completeness of reasoning in each attempt
5. Assess the validity of each attempt's conclusions given their observations

Then:
1. Using the combined observations and analysis from all attempts, determine the most accurate answer. You can either:
   a) Select one of the existing answers if you agree with its reasoning and conclusion
   b) Form your own conclusion if you believe the combined observations support a different answer
2. Output your chosen answer in <answer></answer> tags
3. Follow this with a brief explanation in <explanation></explanation> tags that:
   - If choosing an existing answer: Explain why this attempt's reasoning was most compelling
   - If choosing your own answer: Explain how the combined observations from multiple attempts support your conclusion

Note: 
- You have access to all observations and details mentioned across all three attempts
- Feel free to combine strong reasoning points from multiple attempts
- Focus on connecting concrete observations to logical conclusions
- Don't feel constrained by the existing answers if the combined evidence suggests a different conclusion
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
        Process a batch of attempts and questions concurrently
        
        Args:
            items (List[Dict]): List of dictionaries containing video_id, attempts, and question
            
        Returns:
            List[Dict]: List of results
        """
        tasks = []
        for item in items:
            task = self.reason_over_attempts(
                item['attempts'],
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
            
            # Load the description JSON with multiple attempts
            description_file = description_dir / f"{video_id}_result.json"
            
            if not description_file.is_file():
                logger.error(f"No description JSON found for video ID {video_id} at {description_file}")
                continue
            
            try:
                with open(description_file, 'r') as f:
                    description_data = json.load(f)
                
                attempts = description_data.get("attempts", [])
                
                if not attempts:
                    logger.error(f"No attempts found in {description_file}")
                    continue
                
                items_to_process.append({
                    'video_id': video_id,
                    'attempts': attempts,
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