from flask import Flask, render_template, send_file, request
from flask_socketio import SocketIO
import os
import json
# https://tutorialedge.net/typescript/typescript-socket-io-tutorial/
app = Flask(__name__, template_folder='../../interface', static_folder="../../interface/dist", static_url_path="/dist")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@socketio.on('json')
def handle_json(json):
    print('received json: ' + str(json))


@socketio.on('my event')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
    return 'one', 2


def _check_and_send_file(filename, dirname):
    filename = filename.strip()
    target_filepath = os.path.join(dirname, filename)
    if not os.path.exists(target_filepath):
        print("Could not find ", target_filepath)
        return json.dumps(
            {'status': 'failed', 'msg': 'cannot find the file'})
    return send_file(target_filepath)


# The landing page for user with uid
@app.route('/<uuid>')
def index(uuid):
    return render_template('demo.html', uuid=uuid)

 # Called to access files in the assets folder
@app.route('/assets/<source>', methods=['GET'])
def get_file(source):
    return _check_and_send_file(source, "../../interface/assets")

if __name__ == '__main__':
    socketio.run(app)
