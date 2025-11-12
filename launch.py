"""
Launch script for AI Contract Risk Analyzer
Starts both API and frontend (if available)
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running")
            return True
    except:
        pass
    
    print("✗ Ollama not running. Start with: ollama serve")
    return False

def check_models():
    """Check if required models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        required = "llama3:8b"
        if any(required in name for name in model_names):
            print(f"✓ Model {required} available")
            return True
        else:
            print(f"✗ Model {required} not found. Pull with: ollama pull llama3:8b")
            return False
    except:
        return False

def start_api():
    """Start FastAPI server"""
    print("\n" + "="*60)
    print("Starting FastAPI Server...")
    print("="*60)
    
    subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        response = requests.get("http://localhost:8005/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✓ API Server running at: http://localhost:8005")
            print("✓ Documentation at: http://localhost:8005/api/docs")
            return True
    except:
        pass
    
    print("✗ Failed to start API server")
    return False

def start_frontend():
    """Start frontend server (if available)"""
    if not Path("static/index.html").exists():
        print("\n✗ Frontend not found at static/index.html")
        return False
    
    print("\n" + "="*60)
    print("Starting Frontend Server...")
    print("="*60)
    
    subprocess.Popen([
        sys.executable, "-m", "http.server", "3000",
        "--directory", "static"
    ])
    
    time.sleep(2)
    
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✓ Frontend running at: http://localhost:3000")
            return True
    except:
        pass
    
    print("✗ Failed to start frontend server")
    return False

def main():
    """Main launch function"""
    print("="*60)
    print("AI Contract Risk Analyzer - Launch Script")
    print("="*60)
    
    # Pre-flight checks
    print("\nPre-flight checks:")
    print("-"*60)
    
    ollama_ok = check_ollama()
    models_ok = check_models() if ollama_ok else False
    
    if not ollama_ok:
        print("\n⚠️  Warning: Ollama not running. Some features may not work.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Start services
    api_ok = start_api()
    
    if not api_ok:
        print("\n✗ Failed to start API. Exiting.")
        return
    
    frontend_ok = start_frontend()
    
    # Summary
    print("\n" + "="*60)
    print("Launch Complete!")
    print("="*60)
    print(f"API Server: {'✓' if api_ok else '✗'} http://localhost:8000")
    print(f"API Docs: {'✓' if api_ok else '✗'} http://localhost:8000/api/docs")
    print(f"Frontend: {'✓' if frontend_ok else '✗'} http://localhost:3000")
    print("\nPress Ctrl+C to stop all services")
    print("="*60)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()