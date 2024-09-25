# Enter your code here
import cv2
import numpy as np

# Global variables for user to select a color patch
selected = False
rect = (0, 0, 0, 0)
tolerance = 0

# Callback to adjust the tolerance
def adjust_tolerance(value):
    global tolerance
    tolerance = value

# Function to calculate the range of values
def range_value(min_values, max_values):
    # Apply tolerance adjustment
    min_values1 = np.clip(min_values - tolerance, 0, 255)  # Lower bounds
    max_values1 = np.clip(max_values + tolerance, 0, 255)  # Upper bounds
    
    return min_values1, max_values1

# Mouse callback function to select a rectangular color patch
def select_patch(event, x, y, flags, param):
    global selected, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = np.array([x-30, y-30, x+30, y+30]).astype(dtype=np.uint8)
        selected = True

# Main function to apply chroma keying
def chroma_keying(video_path, background_image_path):
    global min_hsv, max_hsv

    # Open video and background image
    cap = cv2.VideoCapture(video_path)
    background = cv2.imread(background_image_path)
    
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return
    
    # Get video dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read video frame.")
        return
    h, w = frame.shape[:2]

    # Resize background to match video dimensions
    background = cv2.resize(background, (w, h))
    
    # Create window and set mouse callback
    cv2.namedWindow('Select Patch')
    cv2.setMouseCallback('Select Patch', select_patch)
    
    # Display the first frame and let the user select a patch of green screen
    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        # org
        org = (10, 50)
        # fontScale
        fontScale = 1   
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2 
        # Using cv2.putText() method
        frame = cv2.putText(frame, 'click to filter color range', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Select Patch', frame)
        key = cv2.waitKey(1) & 0xFF
        if selected:
            break
        elif key == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            return

    # Extract the selected color range from the patch
    selected_patch = frame[rect[1]:rect[3], rect[0]:rect[2]]
    hsv_patch = cv2.cvtColor(selected_patch, cv2.COLOR_BGR2HSV)
    min_hsv = np.min(hsv_patch, axis=(0, 1))  # Minimum HSV value in the patch
    max_hsv = np.max(hsv_patch, axis=(0, 1))  # Maximum HSV value in the patch

    # Close the patch selection window
    cv2.destroyWindow('Select Patch')

    # Create a window for the output video and the tolerance trackbar
    cv2.namedWindow('Chroma Keyed Video')
    MaxTolerance = 50  # Set a reasonable maximum tolerance value
    cv2.createTrackbar('Tolerance', 'Chroma Keyed Video', tolerance, MaxTolerance, adjust_tolerance)

    # Prepare to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (w, h))

    # Process each frame for chroma keying
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Adjust the HSV range with the current tolerance
        min_hsv_adjusted, max_hsv_adjusted = range_value(min_hsv, max_hsv)

        # Create a mask where the green screen is detected based on adjusted tolerance
        mask = cv2.inRange(hsv_frame, min_hsv_adjusted, max_hsv_adjusted)
        
        # Invert the mask to keep the subject and remove the background
        mask_inv = cv2.bitwise_not(mask)

        # Use the mask to extract the subject
        subject = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Use the inverse mask to extract the background
        background_masked = cv2.bitwise_and(background, background, mask=mask)

        # Combine the subject with the new background
        result = cv2.add(subject, background_masked)

        # Display the result
        cv2.imshow('Chroma Keyed Video', result)
        
        # Write the frame to the output video
        out.write(result)

        # Exit on 'esc' key
        if cv2.waitKey(25) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Path to video and background image
video_path = 'greenscreen-asteroid.mp4'
background_image_path = 'background.jpg'

# Run the chroma keying function
chroma_keying(video_path, background_image_path)
