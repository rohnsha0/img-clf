from flask import Flask, request, jsonify
import utils

app = Flask(__name__)


@app.route('/classification', methods= ['GET', 'POST'])
def classification():
    IMGdata = request.form.get('IMGdata')
    response = jsonify(utils.classify_image(IMGdata))

    response.headers.add('Acess-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print('Starting Python flask server for IMGclassification')
    utils.load_saved_artifacts()
    app.run()
