# Fruit-Freshness-Detection-System
1. Run the application locally from GitHub
    - Prerequisites
      Before starting, make sure the following are installed:
        Python 3.13
        - Check version: python --version
    - Clone the Project
      - Open Terminal (Mac/Linux) or Command Prompt (Windows).
      - Clone the repository: git clone https://github.com/KunrothnyLeyhong/Fruit-Freshness-Detection-System.git
      - Navigate into the project folder: cd Fruit-Freshness-Web
    - Install Dependencies
      - Install all required libraries: pip install -r requirements.txt
      - Expected libraries include:
        - streamlit
        - tensorflow
        - numpy
        - pillow
    - Run the Application
      - Start the Streamlit app: streamlit run app.py
      - You should see output like:
        - Local URL: http://localhost:8501
      - Open that link in your browser.
2. Testing the application using Streamlit Cloud deployment
    - Upload image test
      - Click “Upload fruit image”
      - Upload a fruit image (JPG / PNG):
        - Apple 
        - Banana 
        - Strawberry 
      - Expected behavior
        - Image type
          - Fresh fruit
          - Rotten fruit
        - Expected result
          - 🟢 Fruit is FRESH + high confidence
          - 🔴 Fruit is ROTTEN + high confidence
      The confidence percentage should be close to what Teachable Machine shows.
      No image is saved to the system (no database needed)