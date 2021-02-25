import numpy as np
import cv2

# Config
bw_threshold = 50   # 0-255 
retention_factor = 0.01     # Defines how short the trail of moving objects are  
                            # and how long it takes for new static objects (e.g. recently parked car) to become invisible
video_input = "source.avi"

#open the video file
cap = cv2.VideoCapture(video_input)

# Make first frame the averadge
ret, frame = cap.read()
dim = (852, 480)
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

(B, G, R) = cv2.split(frame.astype("float"))
rAvg = R
bAvg = B
gAvg = G

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    #split bgr
    (B, G, R) = cv2.split(frame.astype("float"))

    # Calculate average
    rAvg = rAvg * (1 - retention_factor) + R * retention_factor
    gAvg = gAvg * (1 - retention_factor) + G * retention_factor
    bAvg = bAvg * (1 - retention_factor) + B * retention_factor

    # merge the average values
    avg = cv2.merge([bAvg, gAvg, rAvg]).astype("uint8")

    # Calculate difference
    difference = cv2.merge([np.absolute(bAvg - B), np.absolute(gAvg - G), np.absolute(rAvg - R)]).astype("uint8")

    # Calculate Output
    output = cv2.cvtColor(cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
    (thresh, output) = cv2.threshold(output, bw_threshold, 255, cv2.THRESH_BINARY)

    # Concatinate frames
    right = cv2.vconcat([frame, avg])
    left = cv2.vconcat([difference, output])
    final = cv2.hconcat([right, left])

    # Text parameters
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    # Add text to video
    cv2.putText(final,'Input', (10, 30), font, fontScale, fontColor, lineType)
    cv2.putText(final,'Average', (10, dim[1] + 30), font, fontScale, fontColor, lineType)
    cv2.putText(final,'Difference', (dim[0] + 10, 30), font, fontScale, fontColor, lineType)
    cv2.putText(final,'Output', (dim[0] + 10, dim[1] + 30), font, fontScale, fontColor, lineType)

    # Display output
    cv2.imshow('Background removal', final)

    # Exit at end of video or when "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean exit
cap.release()
cv2.destroyAllWindows()