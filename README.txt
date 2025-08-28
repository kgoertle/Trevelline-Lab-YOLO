================================
Detection Script Functionality:
================================

	- Allows for either standard detections (--detect), or option for testing (--test-detect).
		This automatically looks for the best.pt file in the appropriate runs / test-runs folder.
	  
	- PI cameras are supported, along with video input.
		Output recordings of processed sources are in associated model logs folder. 
		They are recorded as a timestamp from when the detection first began. (mm-dd-yyyy hh-mm-ss)
		(...\logs\(test / runs)\train (timestamp)\recordings\(usb / picamera / video-input)
	
	- Supports running camera of either type or video input. (--sources picamera0/usb0 \path\to\video.type)
		Independent windows for each source, each with their own FPS, object detection, and generic interaction conters.
		Auto scales windows depending on number of sources, with a supported number of 4.
		Windows are labeled accordingly and can also be adjusted manually like most other programs.
	  
	- Class-to-class interactions are recorded in a .csv.
		Recorded as a timestamp from when the detection first began and ended. (mm-dd hh-mm-ss to mm-dd hh-mm-ss)
		Standard detections save a checkpoint every 1 hour, test detections every minute.
		Records duration of the interaction (in s), how many frames it lasted, and the timestamp from when it began to when it ended.
		A 60 frame temporal buffer to count as an interaction for standard use. (~3 sec at 20 fps)
			15 frame temporal buffer to count as an interaction for test use. (>1 sec at 20 fps)
		(...\logs\(test-runs / runs)\train (timestamp)\interaction-metrics\(usb / picamera / video-input)

	- Adjusted temporal smoothing parameters for bounding boxes.
		Ultralytics default parameters do not use temporal smoothing options.
		The included lab mode (--lab) adjusts these paremeters to be more optimal.
		Can be manually adjusted with arguments. (--smooth (0.0-1.0), --max_history (0-5), --dist-thresh (0-100))
		In-short:
			* smooth adjusts the balance between prioritizing objects on past or current frames
			* max_history adjusts how many frames are stored in smoothing process
			* dist_thresh adjusts smoothing of objects depending on relative past to current position
			
---------------------
Detection Arguments:
---------------------

## To run the detection script with preferred settings:
python detect.py --detect --lab --sources (picamera0, usb0, \path\to\video.type)

## To run the detection script with Ultralytics default settings:
python detect.py --detect --sources (picamera0, usb0, \path\to\video.type)

## To test the detection script with preferred settings:
python detect.py --test-detect --lab --sources (picamera0, usb0, \path\to\video.type)

------------------
Terminal Commands:
------------------

## Run these to get mamba environment setup (after installation):
mama create -n yolo-env python=3.10
mamba activate yolo-env
cd /home/trevelline-lab-pi-1/YOLO

## Required libraries:
numpy>=1.23.0
opencv-python-headless>=4.7.0
tflite-runtime>=2.15.0
picamera2>=0.0.4; sys_platform == "linux"







