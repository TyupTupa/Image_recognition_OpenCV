import cv2

#path for picture to recognize
path = 'reference_image.png'

sift = cv2.SIFT_create(5000)

image = cv2.imread(path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#main function that finds features on frame
def find_features(gray_frame):

    #takes keypoints and descriptors from frame and reference image 
    keypoints_frm, descriptors_frm = sift.detectAndCompute(gray_frame, None)
    keypoints_img, descriptors_img = sift.detectAndCompute(gray_image, None)

    #matcher throws an error when descriptors are none
    if descriptors_frm is None or descriptors_img is None:
        return [], keypoints_frm, keypoints_img
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = matcher.match(descriptors_img, descriptors_frm)

        matches = sorted(matches, key=lambda x: x.distance)
        return matches, keypoints_frm, keypoints_img

#finds similar contours
def find_contours(image):

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, thresh_img = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

#finds coordintes of picture bsed om counturs amd larger match of key points
def find_coordinates(cnts, image, matches, similarity_threshold=5):

    picture_coordinates = {}
    max_similarity_score = 0
    best_coords = None

    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 30:
            x_start, y_start = max(0, x - 15), max(0, y - 15)
            x_end, y_end = min(image.shape[1], x + w + 15), min(image.shape[0], y + h + 15)
            #takes top 10% of all matches
            good_matches = matches[:(int(len(matches) * 0.1))]
            similarity_score = len(good_matches)

            if similarity_score >= similarity_threshold and similarity_score > max_similarity_score:
                max_similarity_score = similarity_score
                best_coords = (x_start, y_start, x_end, y_end)
                print('Similarity score:', max_similarity_score)
    
    if best_coords:
        picture_coordinates['found_object'] = best_coords
    return picture_coordinates

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #makes frame gray for better recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #reshapes image for more beautiful display
    image = cv2.resize(image, (frame.shape[1], frame.shape[0]))

    matches, keypoints_frm, keypoints_img = find_features(gray_frame)
    contours = find_contours(gray_frame)
    location = find_coordinates(contours, gray_frame, matches)

    #displays all rectangles that found on the video frame
    for key, value in location.items():
        rec = cv2.rectangle(frame, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    #shows keypoints connected to reference image
    if matches and keypoints_frm and keypoints_img:
        result_frame = cv2.drawMatches(image, keypoints_img, frame, keypoints_frm, matches[:(int(len(matches) * 0.1))], None, flags=2)
    else:
        result_frame = frame

    cv2.imshow('Picture_recognition', result_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
