
import subprocess
import threading
import time
import os

def run_fastapi():
    os.system(
        "uvicorn api:app "
        "--host 0.0.0.0 --port 8000"
    )

def run_streamlit():
    os.system(
        "streamlit run app.py "
        "--server.port 8501 "
        "--server.address 0.0.0.0 "
        "--server.headless true"
    )

if __name__ == "__main__":
    # Start FastAPI in background thread
    t1 = threading.Thread(target=run_fastapi)
    t1.daemon = True
    t1.start()
    print("FastAPI started on port 8000")
    
    time.sleep(3)
    
    # Start Streamlit in main thread
    print("Streamlit starting on port 8501")
    run_streamlit()
