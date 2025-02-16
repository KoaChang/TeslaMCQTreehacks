import cv2
import os
from pathlib import Path

def extract_frames(video_dir):
    """
    Extract 5 equally spaced frames from each video in the specified directory.
    
    Args:
        video_dir (str): Path to directory containing the videos
    """
    # Create output directory if it doesn't exist
    output_base_dir = Path('extracted_frames')
    output_base_dir.mkdir(exist_ok=True)
    
    # Get all mp4 files in the directory
    video_files = list(Path(video_dir).glob('*.mp4'))
    
    for video_path in video_files:
        try:
            # Open the video
            cap = cv2.VideoCapture(str(video_path))
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames <= 0:
                print(f"Error: Could not read frames from {video_path.name}")
                continue
                
            # Calculate frame indices to extract (5 equally spaced frames)
            frame_indices = [int(i * (total_frames - 1) / 4) for i in range(5)]
            
            # Create directory for this video's frames
            video_name = video_path.stem
            output_dir = output_base_dir / video_name
            output_dir.mkdir(exist_ok=True)
            
            # Extract and save frames
            for frame_num, frame_idx in enumerate(frame_indices, 1):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    print(f"Error: Could not read frame {frame_idx} from {video_path.name}")
                    continue
                
                # Add text to frame
                font = cv2.FONT_HERSHEY_DUPLEX  # Changed to DUPLEX for bolder font
                text = f"frame {frame_num}"
                font_scale = 1.5  # Increased font size
                thickness = 3
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Position text in bottom left with padding
                padding = 20
                x = padding
                y = frame.shape[0] - padding  # padding pixels from bottom
                
                # Draw semi-transparent background rectangle
                bg_rect_pts = [
                    (x - padding//2, y + padding//2),
                    (x + text_width + padding//2, y - text_height - padding//2)
                ]
                
                overlay = frame.copy()
                cv2.rectangle(overlay, bg_rect_pts[0], bg_rect_pts[1], (0, 0, 0), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                
                # Add text with white outline for better visibility
                cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2)  # outline
                cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), thickness)  # main text
                
                # Save frame
                output_path = output_dir / f"frame_{frame_num}.jpg"
                cv2.imwrite(str(output_path), frame)
            
            print(f"Processed {video_path.name}: Extracted {len(frame_indices)} frames")
            
        except Exception as e:
            print(f"Error processing {video_path.name}: {str(e)}")
        finally:
            cap.release()

if __name__ == "__main__":
    video_directory = "videos"  # Change this if your videos are in a different directory
    extract_frames(video_directory)