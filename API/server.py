# app.py
from flask_cors import CORS
from flask import Flask, request, jsonify
app = Flask(__name__)
CORS(app)
#import pipeline

@app.route('/geturl/', methods=['GET'])
def respond():

    # Retrieve the name from url parameter
    name = request.args.get("page", None)
   # result = pipeline.getResult(name)
    # For debugging

    result = {'https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Ring_Style_Combat.jpg/220px-Ring_Style_Combat.jpg': ' Mario stands at the center of a dartboard-like arena divided into four ring-shaped sections with 12 radial slots; each enemy occupies a different slot', 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Kensuke_Tanabe_at_E3.png/180px-Kensuke_Tanabe_at_E3.png': 'Kensuke Tanabe wearing glasses and a red tie'}
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        # response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"
        response = result

    # Return the response in json format
    return jsonify(response)

@app.route('/post/', methods=['POST'])
def post_something():
    result = pipeline.printResults()
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {name} to our awesome platform!!",
            # Add this option to distinct the POST request
            "METHOD" : "POST"
        })
    else:
        return jsonify({
            "ERROR": "no name found, please send a name."
        })

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our server !!</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
