from flask import Blueprint, request, session, redirect, url_for, jsonify

auth_bp = Blueprint("auth", __name__)

USERS = {"micheal": "pass123", "stewart": "pass456"}

@auth_bp.route("/login", methods=["POST"])
def login():
    user = request.form.get("user")
    password = request.form.get("password")
    if user in USERS and USERS[user] == password:
        session["user"] = user
        return redirect(url_for("filter_page"))
    return "<h3>Invalid credentials. Try again.</h3>", 401

@auth_bp.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))
