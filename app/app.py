from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/student_score_predictor.pkl")


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction=None,
        form_data={}
        )


@app.route("/predict", methods=["POST"])
def predict():
    student_data = {

        "Hours_Studied": float(request.form["Hours_Studied"]),

        "Attendance": float(request.form["Attendance"]),

        "Previous_Scores": float(request.form["Previous_Scores"]),

        "Sleep_Hours": float(request.form["Sleep_Hours"]),

        "Tutoring_Sessions": float(request.form["Tutoring_Sessions"]),

        "Physical_Activity": float(request.form["Physical_Activity"]),

        "Parental_Involvement": request.form["Parental_Involvement"],

        "Access_to_Resources": request.form["Access_to_Resources"],

        "Motivation_Level": request.form["Motivation_Level"],

        "Family_Income": request.form["Family_Income"],

        "Teacher_Quality": request.form["Teacher_Quality"],

        "Peer_Influence": request.form["Peer_Influence"],

        "Distance_from_Home": request.form["Distance_from_Home"],

        "Parental_Education_Level": request.form["Parental_Education_Level"],

        "Extracurricular_Activities": request.form["Extracurricular_Activities"],

        "Internet_Access": request.form["Internet_Access"],

        "School_Type": request.form["School_Type"],

        "Learning_Disabilities": request.form["Learning_Disabilities"],

        "Gender": request.form["Gender"]

    }

    input_df = pd.DataFrame([student_data])

    prediction = model.predict(input_df)

    predicted_score = round(prediction[0], 2)

    return render_template(
    "index.html",
    prediction=predicted_score,
    form_data=student_data
    )

if __name__ == "__main__":
    app.run(debug=True)