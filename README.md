
---

# Joint Angle Measurement using OpenPose

This project utilizes **OpenPose** to detect human body keypoints and measure joint angles in real-time using a camera feed. The system calculates the angles between various joints (such as shoulders, elbows, knees, etc.) and visualizes the skeleton along with the calculated angles. This can be useful for applications such as motion analysis, rehabilitation, and ergonomic studies.

The project is based on the **OpenPose** library, developed for human body keypoint detection. It integrates with a webcam feed to perform real-time human pose estimation and angle measurements.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This system captures video from a webcam and processes it using **OpenPose** to detect human body keypoints. After detecting the keypoints, the system calculates the angles between specific joint points, such as the elbow angle, knee angle, and wrist angle, and displays these angles on the video feed.

The angles are calculated using basic geometric principles, specifically using the law of cosines. The calculated angles are then displayed alongside the corresponding body joints.

## Features
- **Real-time Joint Angle Measurement**: Calculates and displays joint angles in real-time.
- **Body Skeleton Visualization**: Displays the human body's skeleton, showing the position of keypoints and the bones connecting them.
- **Multiple Joint Support**: Measures angles for multiple joints, including shoulders, elbows, knees, and ankles.
- **Webcam Compatibility**: Works with any standard webcam for real-time video input.
- **Medical Testing**: Includes scripts for medical testing, such as checking if a person's arm can extend backward to a certain angle.
- **Multiple Projects**: Supports different projects for various medical tests, including single and multi-project tests.

## Installation

To use this project, you'll need to install the required dependencies and configure **OpenPose**. Here's how to set it up:

### Step 1: Clone the repository

```bash
git clone https://github.com/Hollyming/angle_estimition.git
cd angle_estimition
```

### Step 2: Install required Python dependencies

This project requires several Python libraries. You can install them via `pip`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes:
- `opencv-python`: For video capture and image processing.
- `numpy`: For numerical operations.
- `math`: For geometric calculations.

### Step 3: Set up OpenPose

OpenPose is a required dependency for keypoint detection. Follow the installation instructions provided in the [OpenPose GitHub repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to install OpenPose on your system.

Once OpenPose is installed, ensure the following:
- The OpenPose models are downloaded and placed in the directory specified in `params["model_folder"] = "openpose/models/"`.
- Adjust the OpenPose installation path in the code to match where the OpenPose models are stored.

### Step 4: Set up Webcam

This project uses a webcam to capture video. Ensure your webcam is correctly connected and functional. You can change the `cam_idx` parameter in the code if you are using a different camera.

### Step 5: Python Version Compatibility

This project has been tested and works with **Python 3.9**. Compatibility with other Python versions has not been fully tested, so users with other versions should proceed with caution or consider using a virtual environment with Python 3.9.

## Usage

Once the setup is complete, you can run the scripts to start the angle measurement system.

### `angle_15points.py`
- **Description**: Extracts and measures angles using a 15-point skeleton.
- **Key Functionality**:
  - The system detects keypoints of the body (e.g., shoulders, elbows, knees).
  - It calculates the angles between specific joints.
  - It displays the joint angles on the video feed.
- **Exit**:
  - Press `ESC` or `q` to exit the application.

### `angle_25points.py`
- **Description**: Extracts and measures angles using a 25-point skeleton (higher resolution skeleton).
- **Key Functionality**:
  - The system detects 25 keypoints of the body.
  - It calculates angles for more detailed joint movements compared to the 15-point skeleton.
  - Displays joint angles on the video feed for advanced analysis.

### `angle_estimate_UI_single.py`
- **Description**: This is a complete program with a **UI interface** for medical testing, specifically designed to check if a person's arm can extend backward (shoulder extension) to a certain angle.
- **Key Functionality**:
  - Real-time video input with keypoint detection.
  - Measures the backward extension angle of the shoulder.
  - If the arm extends beyond a certain threshold, it indicates a potential issue with shoulder mobility.
  - The program features a simple UI that provides immediate feedback on whether the angle is within the acceptable range.
  - **Exit**: Press `ESC` or `q` to exit the application.

### `angle_estimate_UI_3proj.py`
- **Description**: This is an upgraded version of the medical measurement program, designed to measure and assess three specific arm angles: **shoulder extension**, **shoulder abduction**, and **elbow convergence**.
- **Key Functionality**:
  - The system detects the relevant keypoints for shoulder extension, abduction, and elbow convergence.
  - The measurement process is switched between the three projects by pressing the **space** bar.
  - After completing all measurements, the program outputs the results for all three projects.
  - **Exit**: Press `ESC` or `q` to exit the application.

### Exit
- Press `ESC` or `q` to exit any application.

## Project Structure

```plaintext
joint-angle-measurement/
├── README.md                   # This file
├── angle_15points.py           # 15-point skeleton extraction and angle measurement
├── angle_25points.py           # 25-point skeleton extraction and angle measurement
├── angle_estimate_UI_single.py # UI-based single-project medical testing (shoulder extension)
├── angle_estimate_UI_3proj.py  # UI-based multi-project medical testing (shoulder extension, abduction, elbow)
├── requirements.txt           # Python dependencies
└── openpose/                  # OpenPose installation and model files (external dependency)
```

- **`angle_15points.py`**: This script uses a 15-point skeleton model to detect keypoints and measure angles for basic motion analysis.
- **`angle_25points.py`**: Similar to the 15-point model, but with a 25-point skeleton, providing more detailed keypoint detection for more accurate angle measurements.
- **`angle_estimate_UI_single.py`**: A program with a UI for medical testing, specifically measuring the backward extension of the arm (shoulder extension).
- **`angle_estimate_UI_3proj.py`**: An upgraded medical measurement program with three separate tests for shoulder extension, shoulder abduction, and elbow convergence.
- **`requirements.txt`**: Lists the Python libraries needed for the project.
- **`openpose/`**: Contains the necessary OpenPose files (this folder should be set up according to OpenPose's instructions).

## Contributing

We welcome contributions to improve the project. If you would like to contribute, please fork the repository, create a new branch, and submit a pull request.

### Steps for contributing:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -am 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### Notes:
- Ensure that you have the correct version of **OpenPose** installed, as compatibility issues may arise with different versions of the library.
- This project assumes you're working with a webcam, but it can be adapted for use with recorded video files or other input devices.
- **Python 3.9** is the supported version for this project. If you're using another version, compatibility may not be guaranteed.
