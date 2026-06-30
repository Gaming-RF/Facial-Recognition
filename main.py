import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from dotenv import load_dotenv
    load_dotenv()

    from app import create_app

    api_key = os.getenv("MIMO_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        print("WARNING: MIMO_API_KEY not set in .env — MiMo analysis disabled")
        print("Get yours at: https://platform.xiaomimimo.com\n")

    app = create_app()
    print("Facial Recognition System running at http://localhost:5000")
    print("Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
