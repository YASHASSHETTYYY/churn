from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import yaml
import joblib 

webapp_root = "webapp"
params_path = "params.yaml"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir,template_folder=template_dir)

class  NotANumber(Exception):
    def __init__(self, message="Values entered are not Numerical"):
        self.message = message
        super().__init__(self.message)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

model = None

def predict(data):
    global model
    if model is None:
        config = read_params(params_path)
        model_dir_path = config["model_webapp_dir"]
        model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction 

def validate_input(dict_request):
    for _, val in dict_request.items():
        try:
            val=float(val)
        except Exception as e:
            raise NotANumber
    return True

def form_response(dict_request):
    config = read_params(params_path)
    model_vars = config["raw_data_config"]["model_var"]
    target = config["raw_data_config"]["target"]
    features = [f for f in model_vars if f != target]
    try:
        # Filter input to only include expected features
        # This prevents errors if the form has extra fields (like submit buttons)
        data_to_validate = {k: dict_request[k] for k in features if k in dict_request}
        
        if validate_input(data_to_validate):
            data = [data_to_validate[f] for f in features]
            data = [list(map(float, data))]
            response = predict(data)
            return response
    except NotANumber as e:
        response =  str(e)
        return response 

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if request.form:
                dict_req = dict(request.form)
                response = form_response(dict_req)
                return render_template("index.html", response=response)
        except Exception as e:
            print(e)
            error = {"error": "Something went wrong!! Try again later!"}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)