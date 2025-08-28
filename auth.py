from flask import Blueprint, request, session, redirect, url_for, jsonify
from functools import wraps
import jwt
import datetime

auth_bp = Blueprint("auth", __name__)

USERS = {"micheal": "pass123", "stewart": "pass456"}
JWT_SECRET = "supersecretjwt"

@auth_bp.route("/login", methods=["POST"])
def login():
    user = request.form.get("user")
    password = request.form.get("password")
    if user in USERS and USERS[user] == password:
        # Create JWT token
        token = jwt.encode({
            'user': user,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, JWT_SECRET, algorithm='HS256')
        
        session["user"] = user
        session["token"] = token
        return redirect(url_for("filter_page"))
    return "<h3>Invalid credentials. Try again.</h3>", 401

@auth_bp.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    user = data.get("user")
    password = data.get("password")
    
    if user in USERS and USERS[user] == password:
        token = jwt.encode({
            'user': user,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }, JWT_SECRET, algorithm='HS256')
        
        return jsonify({"token": token, "user": user})
    return jsonify({"error": "Invalid credentials"}), 401

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        
        return f(*args, **kwargs)
    return decorated