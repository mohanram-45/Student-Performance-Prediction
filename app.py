from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    try:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=request.form.get("reading_score"),
            writing_score=request.form.get("writing_score"),
        ).get_data_as_data_frame()
    except ValueError as error:
        return render_template("home.html", error=str(error)), 400
    try:
        result = PredictPipeline().predict(data)[0]
    except FileNotFoundError:
        return render_template("home.html", error="Model unavailable. Run the training command first."), 503
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("home.html", error="Prediction failed. Check the local application log."), 500
    return render_template("home.html", results=round(float(result), 2))


if __name__ == "__main__":
    app.run(host="127.0.0.1")
