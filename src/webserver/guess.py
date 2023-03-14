import os
import random
import uuid
import datetime
import csv
import time

from flask import (Blueprint, Flask, flash, make_response, redirect,
                   render_template, request, send_from_directory, session,
                   url_for)

bp = Blueprint('guess', __name__, url_prefix="/guess")

# Define some constants
IMAGE_DIR = os.path.join(os.getcwd(),'data/50States10K/test')
CATEGORIES = os.listdir(IMAGE_DIR)
CATEGORIES.sort()
RESULTS_FILE = 'human_guess.csv'
HINT = False # Show answer in console


# Keep function data in cache for a certain amount of time
def time_cache(expiry_time=600):
    def decorator(fn):
        def decorated_fn(**args):
            if "cache" not in fn.__dict__.keys():
                fn.__dict__["cache"] = {"data": None, "update": time.time()}
            if fn.__dict__["cache"]["data"] is None or fn.__dict__["cache"]["update"] < time.time() - expiry_time:
                fn.__dict__["cache"]["data"] = fn(**args)
                fn.__dict__["cache"]["update"] = time.time()
            return fn.__dict__["cache"]["data"]
        return decorated_fn
    return decorator


if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        f.write("label,guess,corresponding,filename,uuid,datetime\n")


def ensure_cookies(session):
    if "reussis" not in session:
        session["reussis"] = 0
    if "essayes" not in session:
        session["essayes"] = 0
    if "uuid" not in session:
        session["uuid"] = str(uuid.uuid4()) # Session uuid, we don't need it to be unique but just not too common

def give_image(session): # Store an image path in the session cookie
    category = random.choice(CATEGORIES)
    image = random.choice(os.listdir(os.path.join(IMAGE_DIR, category)))
    session["image_filename"] = f"{category}/{image}"
    if HINT:
        print(category)



# Define the guessing game page
@bp.route('/', methods=['GET', 'POST'])
def guess():
    ensure_cookies(session)
    if request.method == 'POST':
        if "image_filename" in session and "guess" in request.form:
            real_filename = (session["image_filename"].split("/")[0], session["image_filename"].split("/")[1])

            # Check if the guess is correct
            result = (request.form['guess'] == real_filename[0])
            if result:
                session["reussis"] += 1
                flash("C'était la bonne réponse !")
            else:
                flash(f"La bonne réponse était '{real_filename[0]}'")
            session["essayes"] += 1

            # Save the result to file
            with open(RESULTS_FILE, 'a') as f:
                f.write(f'{real_filename[0]},{request.form["guess"]},{result},{real_filename[1]},{session["uuid"]},{datetime.datetime.now()}\n')

        elif "guess" not in request.form:
            flash("Veuillez faire au moins un choix")
            return redirect("/guess")
        else:
            return redirect("/guess")

    # Display the guessing game page
    resp = make_response(render_template('guess.html', CHOICES=CATEGORIES, session=session))
    give_image(session)
    return resp


@bp.route("/image.png")
def get_media():
    if "image_filename" not in session:
        abort(403)
    else:
        real_filename = (session["image_filename"].split("/")[0], session["image_filename"].split("/")[1])
        return send_from_directory(os.path.join(IMAGE_DIR, real_filename[0]), real_filename[1])


@bp.route("/stats")
@time_cache(expiry_time=30) # 30 seconds cache
def statistiques():
    with open(RESULTS_FILE, 'r') as f:
        data = list(csv.DictReader(f))

        
        success = len([row for row in data if row["corresponding"]=="True"])
        total = len(list(data))
        users = {
            "0000-0000-0000-0000": {
                "essais": 1,
                "success": 0,
            }
        }
        for row in data:
            if row["uuid"] not in users.keys():
                users[row["uuid"]] = {
                    "essais": 0,
                    "success": 0,
                }
            users[row["uuid"]]["essais"] += 1
            if row["corresponding"] == "True":
                users[row["uuid"]]["success"] += 1

        max_uuid = "0000-0000-0000-0000"
        for user in users.keys():
            if users[user]["success"]/users[user]["essais"] >= users[max_uuid]["success"]/users[max_uuid]["essais"]:
                max_uuid = user

        return render_template("stats.html", success=success, total=total, users=len(users), max_uuid=max_uuid, max_score=users[max_uuid]["success"]/users[max_uuid]["essais"])