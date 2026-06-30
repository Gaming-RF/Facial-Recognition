from app import create_app

app = create_app()

if __name__ == "__main__":
    print("Starting Facial Recognition System...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
