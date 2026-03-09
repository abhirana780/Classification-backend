import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

print("Initializing HR Analytics Backend...")

# Configure Flask to serve the frontend folder
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app = Flask(__name__, static_folder=frontend_dir, static_url_path='')
CORS(app)

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

# Generate Synthetic HR Datasets with 3D Features and Bias
def generate_hr_data(dataset_type):
    np.random.seed(42)
    data = []
    
    if dataset_type == 'burnout':
        # X: Hours (100-300), Y: Satisfaction (0-1), Z: Salary (40k-150k)
        h1 = np.random.normal(270, 15, 80)
        s1 = np.random.normal(0.2, 0.08, 80)
        sal1 = np.random.normal(120000, 15000, 80)
        b1 = np.random.choice([0, 1], p=[0.9, 0.1], size=80) 
        
        h2 = np.random.normal(160, 15, 80)
        s2 = np.random.normal(0.8, 0.08, 80)
        sal2 = np.random.normal(90000, 15000, 80)
        b2 = np.random.choice([0, 1], p=[0.8, 0.2], size=80)
        
        # Group 3: Low hours, medium sat. BIAS: 90% are returning mothers / parental leave
        h3 = np.random.normal(130, 15, 80)
        s3 = np.random.normal(0.5, 0.08, 80)
        sal3 = np.random.normal(60000, 10000, 80)
        b3 = np.random.choice([0, 1], p=[0.1, 0.9], size=80)
        
        hours = np.concatenate([h1, h2, h3])
        satisfaction = np.concatenate([s1, s2, s3])
        salary = np.concatenate([sal1, sal2, sal3])
        bias = np.concatenate([b1, b2, b3])
        
        for h, s, sal, b in zip(hours, satisfaction, salary, bias):
            data.append({"x": float(h), "y": float(np.clip(s, 0, 1)), "z": float(sal), "bias": int(b)})
            
    elif dataset_type == 'performance':
        # X: Tenure ( yrs), Y: Performance (1-5), Z: Commute Distance (miles)
        t1, p1, d1 = np.random.normal(1.5, 0.5, 80), np.random.normal(2.5, 0.4, 80), np.random.normal(10, 5, 80)
        t2, p2, d2 = np.random.normal(6.0, 1.0, 80), np.random.normal(4.5, 0.4, 80), np.random.normal(15, 5, 80)
        # Group 3: High tenure, low perf. BIAS: 90% are age > 50
        t3, p3, d3 = np.random.normal(8.0, 1.0, 80), np.random.normal(2.0, 0.4, 80), np.random.normal(45, 10, 80) 
        
        b1 = np.random.choice([0, 1], p=[0.5, 0.5], size=80)
        b2 = np.random.choice([0, 1], p=[0.5, 0.5], size=80)
        b3 = np.random.choice([0, 1], p=[0.1, 0.9], size=80)
        
        tenure = np.concatenate([t1, t2, t3])
        perf = np.concatenate([p1, p2, p3])
        dist = np.concatenate([d1, d2, d3])
        bias = np.concatenate([b1, b2, b3])
        
        for t, p, d, b in zip(tenure, perf, dist, bias):
            data.append({"x": float(np.clip(t, 0, 10)), "y": float(np.clip(p, 1, 5)), "z": float(d), "bias": int(b)})
            
    elif dataset_type == 'culture':
        # Non-spherical data to break K-Means (Make Moons)
        # We scale it to look like HR metrics (e.g., Empathy vs Ambition)
        X, Y = make_moons(n_samples=240, noise=0.08, random_state=42)
        # X[:, 0] range is roughly -1 to 2. Let's scale to 1-10
        x_scaled = (X[:, 0] + 1) * 3 + 1
        y_scaled = (X[:, 1] + 1) * 3 + 1
        z = np.random.normal(50, 5, 240) # 3rd dim is just noise or age
        bias = np.random.choice([0, 1], size=240)
        
        for i in range(240):
            data.append({"x": float(x_scaled[i]), "y": float(y_scaled[i]), "z": float(z[i]), "bias": int(bias[i])})
            
    return data

@app.route('/dataset/<name>', methods=['GET'])
def get_dataset(name):
    print(f"Dataset requested: {name}")
    data = generate_hr_data(name)
    df = pd.DataFrame(data)
    
    # Calculate Elbow (inertias for K=1 to 10)
    X_scaled = StandardScaler().fit_transform(df[['x', 'y', 'z']])
    
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        kmeans.fit(X_scaled)
        inertias.append(float(kmeans.inertia_))
    
    print(f"Returning {len(data)} rows and {len(inertias)} elbow points")
    return jsonify({
        "data": data,
        "elbow": inertias
    })

@app.route('/cluster', methods=['POST'])
def run_cluster():
    req = request.json
    k = req.get('k', 3)
    dataset_name = req.get('dataset', 'burnout')
    algo = req.get('algorithm', 'kmeans')
    print(f"Clustering request: dataset={dataset_name}, k={k}, algo={algo}")
    
    data = generate_hr_data(dataset_name)
    df = pd.DataFrame(data)
    
    # We standardize the features before clustering so that Salary (100k) doesn't completely drown out Satisfaction (0.8)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[['x', 'y', 'z']])
    
    centroids = []
    if algo == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled).tolist()
        
        # Unscale centroids back to original values for the UI
        unscaled_centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids = unscaled_centroids.tolist()
    else:
        # DBSCAN Algorithm
        # eps needs to be tuned for standard scaled data (usually around 0.3 - 0.8)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X_scaled).tolist()
        
        # Calculate centroids (mean) of the resulting DBSCAN clusters to show in insights
        for c in set(labels):
            if c != -1:
                cluster_pts = df[['x', 'y', 'z']][np.array(labels) == c]
                centroids.append(cluster_pts.mean().tolist())
    
    labeled_data = []
    for i, row in enumerate(data):
        labeled_data.append({
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "bias": row["bias"],
            "cluster": int(labels[i])
        })
        
    return jsonify({
        "data": labeled_data,
        "centroids": centroids
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
