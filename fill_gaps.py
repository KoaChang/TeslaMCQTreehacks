import os
import csv
import base64
from openai import OpenAI
from pathlib import Path
import logging
from typing import List, Dict
import json
from typing import Optional, Set

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPT4VProcessor:
    def __init__(self, api_key: str = "", video_ids: Optional[Set[str]] = None):
        """
        Initialize the GPT-4V processor with API credentials and optional video IDs filter
        
        Args:
            api_key (str): OpenAI API key
            video_ids (Optional[Set[str]]): Set of video IDs to process. If None, processes all allowed videos
        """
        self.client = OpenAI(api_key=api_key)
        self.video_ids = video_ids

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
        return f"""You have 5 equally spaced frames (Frame 1 through Frame 5) captured from a 5-second dashcam video, taken from the driver's forward-facing perspective.

Using these frames, answer the following multiple-choice question. Incorporate any relevant details observed in the frames (for example, lanes, signage, vehicles, pedestrians, traffic signals, road markings, obstructions) that might help in selecting the correct answer. Consider how details may change across the frames and note that some frames may be more crucial than others.

Steps to follow:
1. **Frame-by-Frame Analysis:** Briefly describe the significant elements you notice in each of the 5 frames (e.g., signs, road markings, obstructions, other vehicles, potential hazards).
2. **Contextual Reasoning:** Integrate the observations from each frame. Think about what is happening over time, which elements are most relevant, and how they connect to the question.
3. **Match to Answer Choices:** Relate your findings to each of the multiple-choice options. Eliminate those that are inconsistent with the visual evidence or standard traffic rules, and select the most appropriate remaining choice.
4. **Provide the Best Answer:** Conclude with the final choice that best matches the situation. Output that choice in `<answer></answer>` tags.

Now, here is the question and its multiple-choice options:

{question}
"""

    def process_question(self, video_id: str, question: str) -> Dict:
        """
        Process a single question with its associated frames
        
        Args:
            video_id (str): ID of the video/question
            question (str): The question to answer
            
        Returns:
            Dict: API response
        """
        try:
            # Skip if video_id is not in the specified set (if a set was provided)
            if self.video_ids is not None and video_id not in self.video_ids:
                logger.info(f"Skipping video ID: {video_id} as it is not in the specified list.")
                return {}

            frame_paths = self.get_frame_paths(video_id)
            if not frame_paths:
                raise FileNotFoundError(f"No frames found for video ID {video_id}")

            content = [{"type": "text", "text": self.create_prompt(question)}]
            for path in frame_paths:
                base64_image = self.encode_image(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=4096,
                temperature=0,
                top_p=0
            )
            return {
                "answer": completion.choices[0].message.content,
                "finish_reason": completion.choices[0].finish_reason,
            }
        except Exception as e:
            logger.error(f"Error processing video ID {video_id}: {str(e)}")
            return {"error": str(e)}

def main():
    """
    Main function to process questions for specified video IDs
    """
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY_KOA_4o')
    if not api_key:
        raise ValueError("OPENAI_API_KEY_KOA_4o environment variable not set")

    # Specify the video IDs you want to process
    # Comment out or modify this line to process different video IDs
    video_ids_to_process = {"00023"}  # Example: only process these IDs

    # Initialize processor with specific video IDs
    processor = GPT4VProcessor(api_key, video_ids=video_ids_to_process)
    
    # Create output directory
    output_dir = Path('gpt4v_results_new_prompt')
    output_dir.mkdir(exist_ok=True)

    # Process questions from CSV
    with open('questions.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row['id']
            
            # Skip if video_id is not in the specified set
            if video_ids_to_process is not None and video_id not in video_ids_to_process:
                logger.info(f"Skipping video ID: {video_id}")
                continue
            
            question = row['question']
            logger.info(f"Processing video ID: {video_id}")
            result = processor.process_question(video_id, question)

            if result:
                output_path = output_dir / f"{video_id}_result.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"Completed processing video ID: {video_id}")

if __name__ == "__main__":
    main()