import os
import sys
import base64
import cv2
import numpy as np
import time
import subprocess # Added import
from VLM_Agent.api_yoloVLM import call_api_vlm, call_api_llm

# ... existing encode_image function ...
def encode_image(image_path):
    """Convert an image to base64 encoding."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
        
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

# ... existing test_llm function ...
def test_llm():
    """Test the text-only LLM function."""
    print("\n===== Testing LLM API =====")
    prompt = "Write a short poem about robots helping humans."
    
    try:
        print(f"Sending prompt: '{prompt}'")
        
        # Measure response time
        start_time = time.time()
        response = call_api_llm(prompt)
        end_time = time.time()
        response_time = end_time - start_time
        
        print("\nResponse from LLM:")
        print(response)
        print(f"\n✅ LLM test completed successfully! Response time: {response_time:.2f} seconds")
    except Exception as e:
        print(f"\n❌ LLM test failed: {str(e)}")


# ... existing test_vlm function ...
def test_vlm(image_path):
    """Test the vision language model function."""
    print("\n===== Testing VLM API =====")
    
    # Encode the image
    encode_start = time.time()
    base64_image = encode_image(image_path)
    encode_time = time.time() - encode_start
    print(f"Image encoding time: {encode_time:.2f} seconds")
    
    if base64_image is None:
        return
    
    # Create a simple prompt
    prompt = "Describe what you see in this image in detail."
    
    try:
        print(f"Sending image from {image_path} with prompt: '{prompt}'")
        
        # Measure response time
        start_time = time.time()
        response = call_api_vlm(prompt, base64_image)
        end_time = time.time()
        response_time = end_time - start_time
        
        print("\nResponse from VLM:")
        print(response)
        print(f"\n✅ VLM test completed successfully! Response time: {response_time:.2f} seconds")
    except Exception as e:
        print(f"\n❌ VLM test failed: {str(e)}")
        print("This could be because the model doesn't support image input.")
        print("Consider using a multimodal model like 'llava' instead of 'gemma3'")

def start_ollama_server():
    """Attempts to start the Ollama server in the background."""
    print("Attempting to start Ollama server...")
    try:
        # Start ollama serve in the background, redirecting stdout/stderr
        # This prevents the script from hanging if the server is already running
        # or if it prints output.
        process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Ollama server process started (or was already running).")
        # Give the server a moment to initialize
        time.sleep(5)
        return process # Return the process object if needed later (e.g., to terminate)
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        return None
    except Exception as e:
        print(f"An error occurred while trying to start Ollama: {e}")
        return None

def main():
    # Attempt to start the Ollama server
    ollama_process = start_ollama_server()
    # Optional: Check if server started successfully if needed
    # if ollama_process is None:
    #     print("Exiting due to Ollama server start failure.")
    #     sys.exit(1)

    total_start_time = time.time()
    print("Testing api_yoloVLM.py functionality")
    
    # Test text-only LLM
    test_llm()
    
    # Test Vision Language Model
    # First, let's create a simple test image if none is specified
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use existing image path instead of creating one
        image_path = "/home/admin/demo_project/dataset/ref_image/track_train_level_0_2.png"
        if not os.path.exists(image_path):
            # Create a simple test image if the path doesn't exist
            image_path = "/tmp/test_image.jpg"
            img = np.ones((300, 300, 3), dtype=np.uint8) * 255  # White background
            cv2.putText(img, "Hello World!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 255), 2)
            cv2.imwrite(image_path, img)
        print(f"Using image at {image_path}")
    
    test_vlm(image_path)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal test execution time: {total_time:.2f} seconds")

    # Optional: Terminate the Ollama server process if it was started by this script
    # if ollama_process:
    #     print("Attempting to stop the Ollama server process...")
    #     ollama_process.terminate()
    #     try:
    #         ollama_process.wait(timeout=5) # Wait a bit for termination
    #         print("Ollama server process stopped.")
    #     except subprocess.TimeoutExpired:
    #         print("Ollama server process did not terminate gracefully, killing.")
    #         ollama_process.kill()


if __name__ == "__main__":
    main()