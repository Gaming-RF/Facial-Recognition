import logging
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    from dotenv import load_dotenv
    load_dotenv()

    from app import create_app

    app = create_app()
    logger = logging.getLogger(__name__)

    api_key = os.getenv("MIMO_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        logger.warning("MIMO_API_KEY not set in .env — MiMo analysis disabled")
        logger.warning("Get yours at: https://platform.xiaomimimo.com")

    logger.info("Facial Recognition System running at http://localhost:5000")
    logger.info("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
