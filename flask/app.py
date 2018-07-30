from flask import Flask

app = Flask(__name__)

@app.route('/')
    def hello():
    return 'Hello'

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('/var/www/uploads/uploaded_file.txt')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
