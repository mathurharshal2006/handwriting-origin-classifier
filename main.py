
import os, subprocess
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])

