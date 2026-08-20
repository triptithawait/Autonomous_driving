# SmartNav AI

Vehicle-size-aware navigation system for Indian road networks, combining computer vision, fuzzy logic, and graph-based route optimization to suggest routes that are safer and more feasible for different vehicle types.

## Overview

Traditional navigation systems primarily optimize for shortest distance or travel time. In urban and semi-urban environments, especially in India, this often leads to unsafe or impossible routes for larger vehicles such as SUVs, buses, or trucks.

SmartNav AI addresses this by:

- estimating road width from road images using a CNN-based classifier,
- evaluating road suitability using fuzzy logic rules,
- optimizing routes with vehicle-aware graph costs, and
- visualizing results through an interactive Streamlit dashboard.

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

### 1. Road Width Analysis
The CNN model categorizes roads into broad classes like narrow, medium, and wide based on uploaded images or camera input.

### 2. Fuzzy Logic Decision System
The fuzzy engine evaluates road suitability against the user's vehicle dimensions and outputs a suitability score from 0 to 100.

### 3. Route Optimization
The graph engine computes both:

- the standard shortest path, and
- the vehicle-aware safe path that avoids unsuitable roads.

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

## License

This project is intended for academic and research use.

## Authors

- Triptithawait

## Repository

- GitHub: https://github.com/triptithawait/Autonomous_driving
