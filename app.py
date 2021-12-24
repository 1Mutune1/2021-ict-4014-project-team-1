from flask import Flask, request, jsonify, render_template, request
from flask_cors import CORS
from csv import DictWriter

import pickle
import numpy as np
from predict import predictML
# from predict import bodyData

app = Flask(__name__)
CORS(app)


@app.route('/<int:id>', methods=['GET'])
def welcome(id):
    return predictML(id)

@app.route('/addrec', methods=["POST"])
def append():
    print(request.json)
    # print(request)
    jsn = request.json
    fieldnames = ['Computer Number','Gender',"Academic Year","Year Of Study","School","Program",'MajorDescription', 'MinorDescription', 'Total number of courses', 'Sponsor', 'Accomodated', 'Moodle logins', 'CA + Exam']
    dict = {"Computer Number":jsn["id"], "Gender":jsn["gender"], "Academic Year":jsn['acyear'],"Year Of Study":jsn["year"],"School":jsn['sch'],"Program":jsn["program"],"MajorDescription":jsn["major"], "MinorDescription":jsn["minor"], "Total number of courses":jsn["totalCourses"], "Sponsor":jsn["sponsor"], "Accomodated":jsn["accomodated"],"Moodle logins":jsn["moodle"], "CA + Exam":jsn["cascore"]}

    with open('Reports Demographics1.csv', 'a') as f_object:
        dw_ob = DictWriter(f_object,fieldnames)
        dw_ob.writerow(dict)
        f_object.close()
    return request.json

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
