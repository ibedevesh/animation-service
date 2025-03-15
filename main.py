from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import uuid
import threading
import json
from together import Together
import subprocess
import logging
from dotenv import load_dotenv
import imageio
import numpy as np
from PIL import Image, ImageDraw

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add this after the imports
load_dotenv()

# Replace the Together API client initialization with:
api_key = os.getenv("TOGETHER_API_KEY", "f0a62b237f7317a9aef6a1ea6ce4a610c9f8c4f6964823b8e632b0decb0c07ed")
client = Together(api_key=api_key)

# In-memory storage for jobs and history
# In a production app, you would use a database
jobs = {}
history = []

# Directory to store generated animations
ANIMATIONS_DIR = "animations"
os.makedirs(ANIMATIONS_DIR, exist_ok=True)

def generate_animation_code(prompt):
    """Generate Python code for animation using Together AI"""
    try:
        logger.info(f"Generating animation code for prompt: {prompt}")
        
        # Create a more specific prompt with example
        messages = [
            {"role": "system", "content": """You are an expert in creating Python animations using the manim library. 
Generate complete, runnable Python code for animations based on text descriptions. 
Here's an example structure:

from manim import *

class CustomAnimation(Scene):
    def construct(self):
        # Create objects
        title = Text("Title")
        title.to_edge(UP)
        
        # Create shapes or custom objects
        circle = Circle(radius=1, color=BLUE)
        square = Square(side_length=2, color=RED)
        
        # Add animations
        self.play(Write(title))
        self.play(Create(circle))
        self.play(Transform(circle, square))
        self.wait()

Make the animation visually appealing with proper timing, transformations, and effects."""},
            {"role": "user", "content": f"Create a detailed manim animation for: {prompt}. Include creative visual elements, smooth transitions, and proper timing. The animation should be engaging and demonstrate the concept clearly. Use color, movement, and text to enhance understanding."}
        ]
        
        # Use a model that's good at code generation
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1,
            stop=[""]
        )
        
        # Extract the generated code
        code = response.choices[0].message.content.strip()
        
        # Clean up the code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        
        # Add necessary imports if they're missing
        if "from manim import *" not in code:
            code = "from manim import *\n" + code
            
        # Verify the code has minimum required elements
        if "class" not in code or "def construct" not in code:
            logger.error("Generated code is missing required elements")
            return generate_fallback_animation(prompt)
            
        logger.info(f"Generated animation code:\n{code}")
        return code
        
    except Exception as e:
        logger.error(f"Error generating animation code: {str(e)}")
        return generate_fallback_animation(prompt)

def generate_fallback_animation(prompt):
    """Generate a fallback animation if the main generation fails"""
    return f"""from manim import *

class FallbackAnimation(Scene):
    def construct(self):
        # Create title
        title = Text("{prompt}", font_size=40)
        title.to_edge(UP)
        
        # Create main objects
        circle = Circle(radius=2, color=BLUE)
        square = Square(side_length=3, color=RED)
        triangle = Triangle(color=GREEN)
        
        # Position objects
        VGroup(circle, square, triangle).arrange(RIGHT, buff=0.5)
        
        # Add animations
        self.play(Write(title))
        self.wait(0.5)
        
        # Create objects with nice effects
        self.play(
            Create(circle),
            GrowFromCenter(square),
            DrawBorderThenFill(triangle),
            run_time=2
        )
        self.wait(0.5)
        
        # Add some movement
        self.play(
            circle.animate.shift(UP*2),
            square.animate.rotate(PI),
            triangle.animate.scale(1.5),
            run_time=2
        )
        self.wait(0.5)
        
        # Add color changes
        self.play(
            circle.animate.set_color(YELLOW),
            square.animate.set_color(PURPLE),
            triangle.animate.set_color(ORANGE),
            run_time=1.5
        )
        self.wait(0.5)
        
        # Final movement
        self.play(
            circle.animate.shift(DOWN*2),
            square.animate.rotate(-PI),
            triangle.animate.scale(0.5),
            run_time=2
        )
        self.wait(1)
"""

def generate_themed_fallback_animation(prompt):
    """Generate a themed fallback animation based on the prompt"""
    prompt_lower = prompt.lower()
    
    if "christmas" in prompt_lower or "tree" in prompt_lower:
        # Christmas tree animation
        frames = []
        width, height = 640, 480
        
        for t in range(60):  # 60 frames
            # Create a new image with a dark blue background (night sky)
            img = Image.new('RGB', (width, height), (10, 10, 40))
            draw = ImageDraw.Draw(img)
            
            # Draw tree trunk
            trunk_coords = [
                (width//2 - 30, height//2 + 100),
                (width//2 + 30, height//2 + 100),
                (width//2 + 30, height),
                (width//2 - 30, height)
            ]
            draw.polygon(trunk_coords, fill=(101, 67, 33))  # Brown
            
            # Draw tree triangles
            for i in range(3):
                triangle_coords = [
                    (width//2, height//2 - 100 - i*80),  # Top
                    (width//2 - 100 + i*20, height//2 + 50 - i*80),  # Bottom left
                    (width//2 + 100 - i*20, height//2 + 50 - i*80)   # Bottom right
                ]
                draw.polygon(triangle_coords, fill=(1, 90, 1))  # Green
            
            # Add twinkling lights
            for i in range(20):
                x = width//2 + int(80 * np.cos(t * 0.1 + i * np.pi/10))
                y = height//2 + int(80 * np.sin(t * 0.1 + i * np.pi/10))
                light_color = (255, 255, 0) if (t + i) % 2 == 0 else (255, 0, 0)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=light_color)
            
            # Add stars in the background
            for _ in range(50):
                star_x = np.random.randint(0, width)
                star_y = np.random.randint(0, height//2)
                brightness = int(128 + 127 * np.sin(t * 0.1 + star_x * star_y))
                draw.point([star_x, star_y], fill=(brightness, brightness, brightness))
            
            # Convert PIL Image to numpy array
            frame = np.array(img)
            frames.append(frame)
        
        # Save animation
        imageio.mimsave('output.mp4', frames, fps=30)
        return None
        
    elif "ball" in prompt_lower or "sphere" in prompt_lower:
        # Bouncing ball animation
        frames = []
        width, height = 640, 480
        
        for t in range(60):
            img = Image.new('RGB', (width, height), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Calculate ball position
            y = int(height//2 + 100 * np.sin(t * 0.1))
            x = width//2
            
            # Draw shadow
            shadow_y = height - 50
            shadow_size = 40 - int(20 * np.sin(t * 0.1))
            draw.ellipse([x-shadow_size, shadow_y-10, x+shadow_size, shadow_y+10],
                        fill=(200, 200, 200))
            
            # Draw ball
            ball_size = 40
            draw.ellipse([x-ball_size, y-ball_size, x+ball_size, y+ball_size],
                        fill=(255, 0, 0))  # Red ball
            
            # Add highlight
            highlight_size = 10
            draw.ellipse([x-highlight_size, y-highlight_size, x+highlight_size, y+highlight_size],
                        fill=(255, 200, 200))
            
            frames.append(np.array(img))
        
        # Save animation
        imageio.mimsave('output.mp4', frames, fps=30)
        return None
    
    else:
        # Default animation with multiple shapes
        frames = []
        width, height = 640, 480
        
        for t in range(60):
            img = Image.new('RGB', (width, height), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # Animated circle
            circle_x = int(width//2 + 100 * np.cos(t * 0.1))
            circle_y = int(height//2 + 100 * np.sin(t * 0.1))
            draw.ellipse([circle_x-30, circle_y-30, circle_x+30, circle_y+30],
                        fill=(0, 0, 255))
            
            # Animated square
            square_x = int(width//2 + 100 * np.cos(t * 0.1 + np.pi))
            square_y = int(height//2 + 100 * np.sin(t * 0.1 + np.pi))
            draw.rectangle([square_x-30, square_y-30, square_x+30, square_y+30],
                        fill=(255, 0, 0))
            
            # Animated triangle
            triangle_x = width//2
            triangle_y = int(height//2 + 50 * np.sin(t * 0.2))
            triangle_coords = [
                (triangle_x, triangle_y - 40),
                (triangle_x - 40, triangle_y + 40),
                (triangle_x + 40, triangle_y + 40)
            ]
            draw.polygon(triangle_coords, fill=(0, 255, 0))
            
            frames.append(np.array(img))
        
        # Save animation
        imageio.mimsave('output.mp4', frames, fps=30)
        return None

def setup_dependencies():
    """Install required system dependencies and Python packages"""
    try:
        # Install system dependencies using Nix
        with open("replit.nix", "w") as f:
            f.write("""
{ pkgs }: {
    deps = [
        pkgs.python39
        pkgs.cairo
        pkgs.pango
        pkgs.pkg-config
        pkgs.ffmpeg
        pkgs.gobject-introspection
        pkgs.glib
        pkgs.gtk3
    ];
}
""")
        
        # Install Python packages
        subprocess.run([
            "pip", "install", 
            "pycairo",
            "pangocairo",
            "manimpango",
            "manim",
            "--no-cache-dir"
        ], check=True)
        
        logger.info("Successfully installed dependencies")
        return True
        
    except Exception as e:
        logger.error(f"Error installing dependencies: {str(e)}")
        return False

def run_animation_code(code, job_id):
    """Run the generated animation code"""
    try:
        # Create a directory for this job
        job_dir = os.path.join(ANIMATIONS_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Save the code to a file
        code_file = os.path.join(job_dir, "animation.py")
        with open(code_file, "w") as f:
            f.write(code)
        
        # Ensure dependencies are installed
        if not hasattr(run_animation_code, '_dependencies_installed'):
            if setup_dependencies():
                run_animation_code._dependencies_installed = True
            else:
                raise Exception("Failed to install dependencies")
        
        # Run the animation code
        process = subprocess.run(
            ["manim", "-pql", "animation.py"],
            capture_output=True,
            text=True,
            cwd=job_dir,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        
        logger.info(f"Animation process stdout: {process.stdout}")
        if process.returncode != 0:
            logger.error(f"Animation generation failed: {process.stderr}")
            raise Exception(f"Animation generation failed: {process.stderr}")
        
        # Find the generated video file
        video_files = [f for f in os.listdir(job_dir) if f.endswith('.mp4')]
        if not video_files:
            raise Exception("No video file was generated")
            
        output_file = os.path.join(job_dir, video_files[0])
        return output_file
        
    except Exception as e:
        logger.error(f"Error running animation code: {str(e)}")
        # Create a placeholder video in case of error
        output_file = os.path.join(job_dir, "output.mp4")
        create_placeholder_video(output_file)
        return output_file

def create_placeholder_video(output_path, prompt=""):
    """Create a placeholder video if generation fails"""
    try:
        # Try to use imageio if available
        try:
            import imageio
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Determine color based on prompt
            color = (0, 0, 200)  # Default blue
            if "red" in prompt.lower():
                color = (200, 0, 0)  # Red
            elif "green" in prompt.lower():
                color = (0, 200, 0)  # Green
            elif "blue" in prompt.lower():
                color = (0, 0, 200)  # Blue
            
            # Create a frame with text
            width, height = 640, 360
            frames = []
            
            for i in range(90):  # 3 seconds at 30fps
                img = Image.new('RGB', (width, height), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)
                
                # Draw a circle that moves
                circle_x = width // 2 + int(100 * np.sin(i * 0.1))
                circle_y = height // 2 + int(50 * np.cos(i * 0.1))
                circle_radius = 50
                draw.ellipse(
                    (circle_x - circle_radius, circle_y - circle_radius, 
                     circle_x + circle_radius, circle_y + circle_radius), 
                    fill=color
                )
                
                # Add text
                draw.text((width//2 - 100, 30), f"Animation for: {prompt}", fill=(255, 255, 255))
                draw.text((width//2 - 100, height - 50), "Placeholder Animation", fill=(255, 255, 255))
                
                frames.append(np.array(img))
            
            # Create a writer object
            writer = imageio.get_writer(output_path, fps=30)
            
            # Write frames
            for frame in frames:
                writer.append_data(frame)
                
            writer.close()
            logger.info(f"Created placeholder video using imageio at {output_path}")
            return
        except Exception as e:
            logger.warning(f"Error creating imageio animation: {str(e)}")
            logger.warning("Falling back to basic file creation")
            
        # If imageio fails, create a simple text file as a placeholder
        with open(output_path, "wb") as f:
            # Create a minimal valid MP4 file (not playable but will download)
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41\x00\x00\x00\x00moov")
        logger.info(f"Created placeholder video file at {output_path}")
    except Exception as e:
        logger.error(f"Error creating placeholder video: {str(e)}")
        # If all else fails, create an empty file
        with open(output_path, "wb") as f:
            f.write(b"")

def process_animation_job(job_id, prompt):
    """Process an animation job in a background thread"""
    try:
        # Update job status
        jobs[job_id]["status"] = "generating_code"
        
        # Clean the prompt to ensure it's just text
        clean_prompt = prompt.strip()
        if clean_prompt.startswith("<think>"):
            clean_prompt = clean_prompt.replace("<think>", "").strip()
        
        # Log the prompt
        logger.info(f"Processing animation for prompt: {clean_prompt}")
        
        # Generate animation code
        code = generate_animation_code(clean_prompt)
        jobs[job_id]["code"] = code
        
        # Update job status
        jobs[job_id]["status"] = "running_code"
        
        # Run the animation code
        output_file = run_animation_code(code, job_id)
        
        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_file"] = output_file
        
        # Add to history
        history.append({
            "id": job_id,
            "prompt": clean_prompt,
            "created_at": int(time.time()),
            "status": "completed"
        })
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.route("/api/animations", methods=["POST"])
def create_animation():
    """Create a new animation job"""
    try:
        data = request.json
        prompt = data.get("prompt")
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Create a new job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "prompt": prompt,
            "status": "queued",
            "created_at": time.time()
        }
        
        # Start processing in a background thread
        threading.Thread(
            target=process_animation_job,
            args=(job_id, prompt)
        ).start()
        
        return jsonify({"job_id": job_id, "status": "queued"}), 201
        
    except Exception as e:
        logger.error(f"Error creating animation: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/animations/<job_id>", methods=["GET"])
def get_animation_status(job_id):
    """Get the status of an animation job"""
    try:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
            
        job = jobs[job_id]
        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "created_at": job["created_at"],
            "error": job.get("error")
        })
        
    except Exception as e:
        logger.error(f"Error getting animation status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/animations/<job_id>/video", methods=["GET"])
def get_animation_video(job_id):
    """Get the generated animation video"""
    try:
        if job_id not in jobs:
            return jsonify({"error": "Job not found"}), 404
            
        job = jobs[job_id]
        
        if job["status"] != "completed":
            return jsonify({"error": "Animation not ready yet"}), 400
            
        output_file = job.get("output_file")
        if not output_file or not os.path.exists(output_file):
            return jsonify({"error": "Video file not found"}), 404
            
        return send_file(output_file, mimetype="video/mp4")
        
    except Exception as e:
        logger.error(f"Error getting animation video: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def get_history():
    """Get animation history"""
    try:
        # Return history in reverse chronological order
        return jsonify(sorted(history, key=lambda x: x["created_at"], reverse=True))
        
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Celetium Animation API is running!"

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)