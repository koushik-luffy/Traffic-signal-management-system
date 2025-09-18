##  Traffic Signal Management System

This project simulates an **AI-powered traffic signal management system** that uses computer vision to detect vehicles and control traffic signals.  
It helps in reducing congestion by adjusting traffic light timing dynamically based on real-time video input.

---

##  Features
- Vehicle detection using **OpenCV** and Haar cascades (`cars.xml`).
- Supports multiple lane video inputs (`laneA.mp4`, `laneB.mp4`, etc.).
- Real-time dashboard (`dashboard.py`) to visualize traffic flow.
- Demo videos included (`demo.mp4`) for testing.
- Can be extended for smart city applications.

---

##  Tech Stack
- **Python 3**
- **OpenCV** (for image & video processing)
- **Streamlit / Tkinter** (for dashboard visualization, optional)
- **NumPy**

---

##  Project Structure
├── cars.xml # Haar cascade for car detection

├── dashboard.py # Main dashboard script

├── opencv.py # Core OpenCV logic for traffic detection

├── demo.mp4 # Sample demo video

├── laneA.mp4 # Lane A test video

├── laneB.mp4 # Lane B test video

├── laneC.mp4 # Lane C test video

├── laneD.mp4 # Lane D test video

└── README.md # Documentation


---

##  Getting Started

### Prerequisites
- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Install dependencies:
```bash
pip install opencv-python numpy streamlit


# To run the dashboard
python dashboard.py

# Or run the OpenCV traffic detection script
python opencv.py

```

##  Demo

[![Watch the demo](https://img.youtube.com/vi/sLbDi59argg/0.jpg)](https://youtu.be/sLbDi59argg)

[![Watch the demo](https://img.youtube.com/vi/sLbDi59argg/0.jpg)](https://youtu.be/nzNhDcnFQPw)





