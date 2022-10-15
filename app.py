import pandas as pd
import numpy as pd
import joblib
import traceback
from flask import Flask, request, jsonify

app=Flask(__name__)

lr=joblib.load("model.pkl")
print("model loaded")


@app.route("/iris", methods=["GET","POST"])

def predict():
	if lr:
		try:
			json_ = request.json
			
		
			prediction=list(lr.predict(json_))
			return jsonify({"Prediction ": str(prediction)})	
				

		except:
			return jsonify({"trace ": traceback.format_exc()})
	else:
		print("first train the model")
		return ("no model is here to use")

if __name__ == "__main__":
	app.run(debug= True)