from flask import Flask
import webbrowser, threading
def run_dashboard():
    app=Flask(__name__)
    @app.route('/')
    def home(): return '<h1>RedditVideoBot 4.0 Dashboard</h1>'
    threading.Timer(1,lambda:webbrowser.open('http://localhost:5000')).start()
    app.run(port=5000)
