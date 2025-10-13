# Hosts the live video footage from the Raspberry Pi to a web interface
from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def index():
    return "Live video feed will be here."

app.run(port=5000)