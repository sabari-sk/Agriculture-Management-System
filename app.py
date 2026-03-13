from flask import Flask, render_template, request
import random, os
from functions import (
	get_crop_recommendation,
	get_fertilizer_recommendation,
	get_crop_health_prediction,
	soil_types,
	Crop_types,
)


app = Flask(__name__)
random.seed(0)
app.config['SECRET_KEY'] = os.urandom(24)

@app.route('/', methods=['GET', 'POST'])
def index():
	return render_template('index.html')

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
	if request.method == "POST":
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = get_crop_recommendation(to_predict_list)
		return render_template("recommend_result.html", result=result)
	else:
		return render_template('crop-recommend.html')

@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
	if request.method == "POST":
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = get_fertilizer_recommendation(
			num_features=to_predict_list[:-2],
			cat_features=to_predict_list[-2:]
		)
		return render_template("recommend_result.html", result=result)
	else:
		return render_template(
			'fertilizer-recommend.html', 
			soil_types=enumerate(soil_types),
			crop_types=enumerate(Crop_types)
		)

@app.route('/crop-health', methods=['GET', 'POST'])
def crop_health():
	if request.method == "POST":
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(float, to_predict_list))
		result = get_crop_health_prediction(to_predict_list)
		return render_template("recommend_result.html", result=result)
	else:
		return render_template('crop-health.html')

if __name__ == '__main__':
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port, debug=False)