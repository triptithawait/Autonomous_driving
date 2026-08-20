# SmartNav AI

Live app: https://vehicle-size-aware-smart-navigation.streamlit.app/

Vehicle-size-aware navigation system for Indian road networks, combining computer vision, fuzzy logic, and graph-based route optimization to suggest routes that are safer and more feasible for different vehicle types.

## Overview

Traditional navigation systems primarily optimize for shortest distance or travel time. In urban and semi-urban environments, especially in India, this often leads to unsafe or impossible routes for larger vehicles such as SUVs, buses, or trucks.

SmartNav AI is a multi-stage perception-to-decision system. It combines:

- perception: CNN-based road-width estimation from camera or image input,
- decision logic: fuzzy rules that translate road widths and vehicle dimensions into suitability scores,
- routing: Dijkstra-based path optimization that avoids roads that are too narrow for the selected vehicle.

This creates a full AI pipeline from sensor input to route recommendation, which is aligned with real-world edge-AI engineering work because it emphasizes model efficiency, inference performance, and deployment constraints instead of raw model complexity alone.

## Key Features

- Vehicle-specific route planning based on vehicle width
- Simulated road network comparison against standard shortest-path routing
- Real-world map integration using OpenStreetMap data
- CNN-based road width estimation from uploaded or captured road images
- Fuzzy-logic suitability scoring for road feasibility
- Interactive UI with Streamlit and Folium

## Project Architecture

```text
SmartNav AI
├── app.py                     # Streamlit application entry point
├── core/
│   ├── graph_engine.py        # Road network and route optimization logic
│   └── osm_engine.py          # OpenStreetMap data and map-based routing
├── models/
│   ├── cnn_model.py           # CNN-based road width classification model
│   └── fuzzy_logic.py         # Fuzzy inference system for suitability scoring
├── report.md                 # Technical report / dissertation content
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore rules
└── assets / sample files     # Supporting image assets
```

## Tech Stack

- Python 3
- Streamlit
- OpenCV
- PyTorch
- NetworkX
- OSMnx
- Folium
- scikit-fuzzy
- GeoPy
- Matplotlib
- NumPy / Pandas

## Installation

1. Clone the repository:

```bash
git clone https://github.com/triptithawait/Autonomous_driving.git
cd Autonomous_driving
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the application with:

```bash
streamlit run app.py
```

Then open the local URL displayed by Streamlit in your browser.

## How It Works

### 1. Perception Layer: Road Width Estimation
The CNN model categorizes road conditions into classes such as narrow, medium, and wide based on uploaded images or live camera input. In the optimization workflow, this model becomes the inference stage to be measured and compressed for edge deployment.

### 2. Decision Layer: Fuzzy Logic
The fuzzy engine evaluates road suitability against the user's vehicle dimensions and outputs a suitability score from 0 to 100. This layer translates perception output into a route-feasibility decision for different vehicle types.

### 3. Routing Layer: Dijkstra Optimization
The graph engine computes both:

- the standard shortest path, and
- the vehicle-aware safe path that avoids unsuitable roads.

This forms a complete perception-to-decision engine: image/video input -> width estimate -> suitability -> route recommendation.

## Model Optimization & Benchmarking

The project includes a TensorFlow/Keras optimization workflow that is designed for edge-AI and deployment interviews. The Phase 1 benchmark script compares the following MobileNetV2 variants:

- baseline FP32 model
- TFLite FP32
- TFLite dynamic-range INT8
- TFLite full-integer INT8
- pruned + quantized variant

The benchmark script measures:

- accuracy
- model size in MB
- average inference latency in ms per image

The scripts generate:

- a CSV summary table under `optimization/results/phase1_benchmark_results.csv`
- a chart under `optimization/results/phase1_benchmark_chart.png`

The Phase 2 live-camera workflow reads from a webcam and measures FPS using the fastest optimized model variant. The Phase 3 efficiency comparison scaffold compares MobileNetV2 against a heavier architecture such as ResNet50 on the same road-width task, focusing on the efficiency tradeoff for edge deployment.

Important note: the repository does not contain the trained model artifact or the road-image dataset needed to generate real benchmark numbers. The scripts were implemented to fail clearly and honestly when those files are missing, rather than to fabricate metrics.

## Usage

The app contains three main tabs:

- Simulated Network: compare AI-optimized routing against standard GPS routing
- Real-World Map: fetch map data and generate route overlays using OSM data
- CNN Width Analysis: upload an image and assess estimated road width and suitability

## Example Scenario

A compact car may navigate a shorter route through a narrow lane, while a truck or SUV may be redirected to a wider and safer route. The system is designed to keep the route feasible for the specific vehicle dimensions rather than blindly following the shortest distance.

## Future Enhancements

- Height-aware planning for large vehicles and bridges
- Real-time congestion integration
- User-reported road condition feedback
- Better model calibration and wider training data
- Full edge deployment on embedded or mobile inference hardware

## Data / Model Requirements Before Full Benchmarking

The following artifacts are required before the optimization, FPS comparison, or efficiency scripts can produce real numbers:

- a trained Keras model file such as `.keras`, `.h5`, or `.hdf5`
- a road image dataset under a `data/`, `dataset/`, or `images/` folder
- optional: a ResNet50 reference model for Phase 3 comparison

Without those files, the scripts exit with a clear message instead of reporting fabricated values.

## License

This project is intended for academic and research use.

## Authors

- Triptithawait

## Repository

- GitHub: https://github.com/triptithawait/Autonomous_driving
