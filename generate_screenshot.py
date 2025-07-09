"""
Screenshot Generator for Stock Price Predictor
This script helps generate a screenshot of the app for the README.
"""

import subprocess
import time
import os

def main():
    print("📸 Generating screenshot for Stock Price Predictor...")
    print("This will start the app and you can take a screenshot manually.")
    print("\nInstructions:")
    print("1. The app will open in your browser")
    print("2. Wait for it to load completely")
    print("3. Take a screenshot of the main page")
    print("4. Save it as 'screenshot.png' in the project root")
    print("5. Close the browser and press Ctrl+C to stop the server")
    
    try:
        # Start the Streamlit app
        print("\n🚀 Starting Streamlit app...")
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n✅ Screenshot generation completed!")
        print("Make sure to save your screenshot as 'screenshot.png'")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")

if __name__ == "__main__":
    main() 