
import os, subprocess
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    subprocess.run([
        "streamlit", "run", "app.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ])
```
```
4. Commit changes
```

---
```
After all 3 steps:
Render auto redeploys!
Wait 5-7 mins!
App will work! 🎉
