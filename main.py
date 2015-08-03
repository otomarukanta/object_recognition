import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    bfmatcher = cv2.BFMatcher()

    retval, frame = cap.read()

    # reference image
    ref_img = cv2.imread("./itiokunin.jpg")
    sift = cv2.xfeatures2d.SIFT_create()
    ref_keypoints, ref_descriptors = sift.detectAndCompute(ref_img, None)

    while True:
        retval, frame = cap.read()

        img = cv2.resize(frame, (960, 540))

        key = cv2.waitKey(1)
        if key == ord("q"):
                break

        keypoints, descriptors = sift.detectAndCompute(img, None)
        matches = bfmatcher.knnMatch(descriptors, ref_descriptors, k=2)

        matches = [m for m, n in matches if m.distance < n.distance * 0.7]

        img = cv2.drawMatches(img, keypoints, ref_img, ref_keypoints,
                              matches, None,
                              matchColor=(255, 0, 0),
                              singlePointColor=(0, 255, 0))

        cv2.imshow("image", img)
