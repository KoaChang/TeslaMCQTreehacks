import os
import csv
import base64
import asyncio
import aiohttp
from openai import AsyncOpenAI
from pathlib import Path
import logging
from typing import List, Dict
import json
from asyncio import Semaphore
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncGPT4VProcessor:
    def __init__(self, api_key: str = "", max_concurrent_requests: int = 5):
        """
        Initialize the GPT-4V processor with API credentials
        
        Args:
            api_key (str): OpenAI API key
            max_concurrent_requests (int): Maximum number of concurrent API requests
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = Semaphore(max_concurrent_requests)

    def encode_image(self, image_path: str) -> str:
        """
        Encode an image file to base64
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_frame_paths(self, video_id: str) -> List[str]:
        """
        Get paths for all frames of a specific video
        
        Args:
            video_id (str): ID of the video/question
            
        Returns:
            List[str]: List of frame image paths
        """
        frame_dir = Path('extracted_frames') / str(video_id)
        frame_paths = sorted(list(frame_dir.glob('frame_*.jpg')))
        return [str(path) for path in frame_paths]

    def create_prompt(self, question: str, frame_count: int = 5) -> str:
        """
        Create the prompt for GPT-4V
        
        Args:
            question (str): The question to answer
            frame_count (int): Number of frames
            
        Returns:
            str: Formatted prompt
        """
        return f"""You have 5 equally spaced frames (Frame 1 through Frame 5) captured from a 5-second dashcam video, taken from the driver’s forward-facing perspective.

Using these frames, answer the following multiple-choice question by choosing the single best answer. Incorporate any relevant details observed in the frames (for example, lanes, signage, vehicles, pedestrians, traffic signals, road markings, obstructions) that might help in selecting the correct answer. Consider how details may change across the frames and note that some frames may be more crucial than others. Explain your reasoning in detail.

Steps to follow:
1. **Frame-by-Frame Analysis:** Describe the significant elements you notice in each of the 5 frames (e.g., signs, road markings, obstructions, other vehicles, potential hazards).
2. **Contextual Reasoning:** Integrate the observations from each frame. Think about what is happening over time, which elements are most relevant, and how they connect to the question.
3. **Match to Answer Choices:** Relate your findings to each of the multiple-choice options. Eliminate those that are inconsistent with the visual evidence or standard traffic rules, and select the most appropriate remaining choice.
4. **Provide the Best Answer:** Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.

Now, here is the question and its multiple-choice options:

{question}
"""

    async def process_question(self, video_id: str, question: str) -> Dict:
        """
        Process a single question with its associated frames
        
        Args:
            video_id (str): ID of the video/question
            question (str): The question to answer
            
        Returns:
            Dict: API response
        """
        async with self.semaphore:  # Limit concurrent requests
            try:
                # Get frame paths
                frame_paths = self.get_frame_paths(video_id)
                
                if not frame_paths:
                    raise FileNotFoundError(f"No frames found for video ID {video_id}")

                # Prepare the content list with the initial text prompt
                content = [
                    {
                        "type": "text",
                        "text": self.create_prompt(question)
                    }
                ]
                
                # Add each frame as an image_url
                for path in frame_paths:
                    base64_image = self.encode_image(path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })

                # Make API request
                completion = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=8192,
                    temperature=0.7,
                )
                
                # Extract the response content
                response = {
                    "video_id": video_id,
                    "answer": completion.choices[0].message.content,
                    "finish_reason": completion.choices[0].finish_reason,
                }
                
                return response

            except Exception as e:
                logger.error(f"Error processing video ID {video_id}: {str(e)}")
                return {
                    "video_id": video_id,
                    "error": str(e)
                }

    async def process_batch(self, questions: List[Dict]) -> List[Dict]:
        """
        Process a batch of questions concurrently
        
        Args:
            questions (List[Dict]): List of dictionaries containing video_id and question
            
        Returns:
            List[Dict]: List of results
        """
        tasks = []
        for q in questions:
            task = self.process_question(q['id'], q['question'])
            tasks.append(task)
        
        return await asyncio.gather(*tasks)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Load API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY_KOA_4o')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_KOA_4o environment variable not set")

    # Initialize processor with concurrent request limit
    processor = AsyncGPT4VProcessor(api_key, max_concurrent_requests=5)

    # Create output directory for results
    output_dir = Path('initial_answers')
    output_dir.mkdir(exist_ok=True)

    # Read all questions from CSV
    questions = []
    with open('questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        questions = list(reader)

    # Process each question 3 times
    logger.info(f"Processing {len(questions)} questions with 3 attempts each...")
    
    all_results = []
    for question in questions:
        # Create 3 tasks for the same question
        tasks = []
        for attempt in range(3):
            task = processor.process_question(question['id'], question['question'])
            tasks.append(task)
        
        # Wait for all 3 attempts to complete
        question_results = await asyncio.gather(*tasks)
        
        # Combine results for this video ID
        video_id = question_results[0]['video_id']
        combined_result = {
            'video_id': video_id,
            'attempts': []
        }
        
        # Add each attempt's result
        for i, result in enumerate(question_results, 1):
            attempt_data = {
                'attempt_number': i,
                'answer': result.get('answer'),
                'finish_reason': result.get('finish_reason'),
                'error': result.get('error')  # Include error if present
            }
            combined_result['attempts'].append(attempt_data)
        
        all_results.append(combined_result)
        
        # Save individual result immediately
        output_path = output_dir / f"{video_id}_result.json"
        with open(output_path, 'w') as f:
            json.dump(combined_result, f, indent=2)
        
        logger.info(f"Completed processing video ID: {video_id} (3 attempts)")

if __name__ == "__main__":
    asyncio.run(main())