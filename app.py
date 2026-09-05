import os
import csv
import copy
import argparse
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model",
    "keypoint_classifier"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "keypoint_classifier.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "keypoint_classifier_label.csv"
)

CSV_PATH = os.path.join(
    MODEL_DIR,
    "keypoint.csv"
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "model",
    "dataset",
    "dataset 1"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)


# ============================================================
# ARGUMENTS
# ============================================================

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0
    )

    parser.add_argument(
        "--width",
        help="cap width",
        type=int,
        default=960
    )

    parser.add_argument(
        "--height",
        help="cap height",
        type=int,
        default=540
    )

    parser.add_argument(
        "--use_static_image_mode",
        action="store_true"
    )

    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    return args


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Argument parsing
    # --------------------------------------------------------

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True


    # --------------------------------------------------------
    # Check important files
    # --------------------------------------------------------

    print()
    print("==============================================")
    print(" SIGN LANGUAGE RECOGNITION SYSTEM")
    print("==============================================")
    print()

    print("Project directory:")
    print(BASE_DIR)
    print()

    print("Checking required files...")
    print()


    # Check model

    if not os.path.isfile(MODEL_PATH):

        print("ERROR: Model file not found!")
        print(MODEL_PATH)

        return


    print("OK - Model found:")
    print(MODEL_PATH)


    # Check labels

    if not os.path.isfile(LABEL_PATH):

        print("ERROR: Label file not found!")
        print(LABEL_PATH)

        return


    print("OK - Label file found:")
    print(LABEL_PATH)


    # Check dataset

    if not os.path.isdir(DATASET_DIR):

        print()
        print("WARNING: Dataset folder not found:")
        print(DATASET_DIR)
        print()
        print("Camera recognition can still work.")
        print("Dataset capture mode will not work.")
        print()


    else:

        print("OK - Dataset folder found:")
        print(DATASET_DIR)


    print()
    print("Loading model...")
    print()


    # --------------------------------------------------------
    # Camera preparation
    # --------------------------------------------------------

    cap = cv.VideoCapture(cap_device)

    cap.set(
        cv.CAP_PROP_FRAME_WIDTH,
        cap_width
    )

    cap.set(
        cv.CAP_PROP_FRAME_HEIGHT,
        cap_height
    )


    if not cap.isOpened():

        print()
        print("ERROR: Could not open camera.")
        print()
        print("Try:")
        print("1. Check that your camera is connected.")
        print("2. Close other applications using the camera.")
        print("3. Try --device 1 if you have multiple cameras.")
        print()

        return


    # --------------------------------------------------------
    # MediaPipe Hands
    # --------------------------------------------------------

    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(

        static_image_mode=use_static_image_mode,

        max_num_hands=2,

        min_detection_confidence=min_detection_confidence,

        min_tracking_confidence=min_tracking_confidence
    )


    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    try:

        keypoint_classifier = KeyPointClassifier(
            model_path=MODEL_PATH
        )

    except Exception as e:

        print()
        print("ERROR while loading the classifier:")
        print(e)
        print()

        cap.release()

        return


    # --------------------------------------------------------
    # Read labels
    # --------------------------------------------------------

    try:

        with open(
            LABEL_PATH,
            encoding="utf-8-sig"
        ) as f:

            keypoint_classifier_labels = csv.reader(f)

            keypoint_classifier_labels = [
                row[0]
                for row in keypoint_classifier_labels
                if len(row) > 0
            ]

    except Exception as e:

        print()
        print("ERROR while reading label file:")
        print(e)
        print()

        cap.release()

        return


    print()
    print("Labels loaded:")
    print(len(keypoint_classifier_labels))
    print()

    print("Camera started successfully.")
    print()
    print("Controls:")
    print("N = Recognition Mode")
    print("K = Capture landmarks from camera")
    print("D = Capture landmarks from dataset")
    print("ESC = Exit")
    print()


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cvFpsCalc = CvFpsCalc(buffer_len=10)


    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    mode = 0


    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        fps = cvFpsCalc.get()


        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv.waitKey(10)

        if key == 27:

            break


        number, mode = select_mode(
            key,
            mode
        )


        # ----------------------------------------------------
        # Camera capture
        # ----------------------------------------------------

        ret, image = cap.read()

        if not ret:

            print("ERROR: Could not read frame from camera.")

            break


        # Mirror image

        image = cv.flip(
            image,
            1
        )

        debug_image = copy.deepcopy(image)


        # ----------------------------------------------------
        # MediaPipe processing
        # ----------------------------------------------------

        rgb_image = cv.cvtColor(
            image,
            cv.COLOR_BGR2RGB
        )

        rgb_image.flags.writeable = False

        results = hands.process(
            rgb_image
        )

        rgb_image.flags.writeable = True


        # ====================================================
        # DATASET MODE
        # ====================================================

        if mode == 2:

            loading_path = os.path.join(
                ASSETS_DIR,
                "om606.png"
            )

            loading_img = cv.imread(
                loading_path,
                cv.IMREAD_COLOR
            )


            # If loading image doesn't exist,
            # create a simple screen.

            if loading_img is None:

                loading_img = np.zeros(
                    (540, 960, 3),
                    dtype=np.uint8
                )


            cv.putText(

                loading_img,

                "Loading...",

                (20, 50),

                cv.FONT_HERSHEY_SIMPLEX,

                1.0,

                (255, 255, 255),

                4,

                cv.LINE_AA
            )


            cv.imshow(
                "Hand Gesture Recognition",
                loading_img
            )


            cv.waitKey(1000)


            # Check dataset

            if not os.path.isdir(DATASET_DIR):

                print()
                print("Dataset folder does not exist:")
                print(DATASET_DIR)
                print()

                mode = 1

                continue


            # ------------------------------------------------
            # Loop through dataset classes
            # ------------------------------------------------

            imglabel = -1


            try:

                class_folders = sorted(
                    os.listdir(DATASET_DIR)
                )

            except Exception as e:

                print("Could not read dataset:", e)

                mode = 1

                continue


            for imgclass in class_folders:

                class_path = os.path.join(
                    DATASET_DIR,
                    imgclass
                )


                if not os.path.isdir(class_path):

                    continue


                imglabel += 1

                numofimgs = 0


                try:

                    image_files = os.listdir(
                        class_path
                    )

                except Exception:

                    continue


                for img_filename in image_files:

                    imgpath = os.path.join(
                        class_path,
                        img_filename
                    )


                    # Skip non-image files

                    if not os.path.isfile(imgpath):

                        continue


                    try:

                        img = cv.imread(
                            imgpath
                        )


                        if img is None:

                            print(
                                "Could not read:",
                                imgpath
                            )

                            continue


                        debug_img = copy.deepcopy(
                            img
                        )


                        # Convert to RGB for MediaPipe

                        img_rgb = cv.cvtColor(
                            img,
                            cv.COLOR_BGR2RGB
                        )


                        img_rgb.flags.writeable = False

                        dataset_results = hands.process(
                            img_rgb
                        )

                        img_rgb.flags.writeable = True


                        if (
                            dataset_results.multi_hand_landmarks
                            is not None
                        ):

                            for hand_landmarks, handedness in zip(

                                dataset_results.multi_hand_landmarks,

                                dataset_results.multi_handedness

                            ):


                                # Bounding box

                                brect = calc_bounding_rect(

                                    debug_img,

                                    hand_landmarks

                                )


                                # Landmarks

                                landmark_list = calc_landmark_list(

                                    debug_img,

                                    hand_landmarks

                                )


                                # Pre-processing

                                pre_processed_landmark_list = (
                                    pre_process_landmark(
                                        landmark_list
                                    )
                                )


                                # Save to CSV

                                logging_csv(

                                    imglabel,

                                    mode,

                                    pre_processed_landmark_list

                                )


                        numofimgs += 1


                    except Exception as e:

                        print(
                            f"Issue with image {imgpath}: {e}"
                        )


                print(
                    f"Num of image of the class "
                    f"{imglabel} is : {numofimgs}"
                )


            mode = 1

            print()
            print("End of dataset job!")
            print()


        # ====================================================
        # NORMAL RECOGNITION MODE
        # ====================================================

        else:

            if results.multi_hand_landmarks is not None:

                for hand_landmarks, handedness in zip(

                    results.multi_hand_landmarks,

                    results.multi_handedness

                ):


                    # ----------------------------------------
                    # Bounding box
                    # ----------------------------------------

                    brect = calc_bounding_rect(

                        debug_image,

                        hand_landmarks

                    )


                    # ----------------------------------------
                    # Landmark calculation
                    # ----------------------------------------

                    landmark_list = calc_landmark_list(

                        debug_image,

                        hand_landmarks

                    )


                    # ----------------------------------------
                    # Pre-processing
                    # ----------------------------------------

                    pre_processed_landmark_list = (
                        pre_process_landmark(
                            landmark_list
                        )
                    )


                    # ----------------------------------------
                    # Save camera landmarks
                    # ----------------------------------------

                    logging_csv(

                        number,

                        mode,

                        pre_processed_landmark_list

                    )


                    # ----------------------------------------
                    # Hand sign classification
                    # ----------------------------------------

                    try:

                        hand_sign_id = keypoint_classifier(
                            pre_processed_landmark_list
                        )

                    except Exception as e:

                        print(
                            "Classification error:",
                            e
                        )

                        hand_sign_id = -1


                    # ----------------------------------------
                    # Finger gesture classification
                    # ----------------------------------------

                    finger_gesture_id = 0


                    # ----------------------------------------
                    # Get label
                    # ----------------------------------------

                    if (

                        0 <= hand_sign_id
                        < len(keypoint_classifier_labels)

                    ):

                        hand_sign_text = (
                            keypoint_classifier_labels[
                                hand_sign_id
                            ]
                        )

                    else:

                        hand_sign_text = "Unknown"


                    # ----------------------------------------
                    # Drawing
                    # ----------------------------------------

                    debug_image = draw_bounding_rect(

                        use_brect,

                        debug_image,

                        brect

                    )


                    debug_image = draw_landmarks(

                        debug_image,

                        landmark_list

                    )


                    debug_image = draw_info_text(

                        debug_image,

                        brect,

                        handedness,

                        hand_sign_text

                    )


            # --------------------------------------------
            # FPS and information
            # --------------------------------------------

            debug_image = draw_info(

                debug_image,

                fps,

                mode,

                number

            )


            # --------------------------------------------
            # Display
            # --------------------------------------------

            cv.imshow(

                "Hand Gesture Recognition",

                debug_image

            )


    # ========================================================
    # RELEASE
    # ========================================================

    cap.release()

    hands.close()

    cv.destroyAllWindows()


# ============================================================
# SELECT MODE
# ============================================================

def select_mode(key, mode):

    number = -1


    # A-Z

    if 65 <= key <= 90:

        number = key - 65


    # n = Recognition

    if key == ord("n"):

        mode = 0


    # k = Camera landmark capture

    if key == ord("k"):

        mode = 1


    # d = Dataset landmark capture

    if key == ord("d"):

        mode = 2


    return number, mode


# ============================================================
# BOUNDING RECTANGLE
# ============================================================

def calc_bounding_rect(image, landmarks):

    image_width = image.shape[1]

    image_height = image.shape[0]


    landmark_array = np.empty(
        (0, 2),
        int
    )


    for landmark in landmarks.landmark:

        landmark_x = min(

            int(landmark.x * image_width),

            image_width - 1

        )


        landmark_y = min(

            int(landmark.y * image_height),

            image_height - 1

        )


        landmark_point = np.array(
            [(landmark_x, landmark_y)]
        )


        landmark_array = np.append(

            landmark_array,

            landmark_point,

            axis=0

        )


    x, y, w, h = cv.boundingRect(
        landmark_array
    )


    return [
        x,
        y,
        x + w,
        y + h
    ]


# ============================================================
# LANDMARK LIST
# ============================================================

def calc_landmark_list(image, landmarks):

    image_width = image.shape[1]

    image_height = image.shape[0]


    landmark_point = []


    for landmark in landmarks.landmark:

        landmark_x = min(

            int(landmark.x * image_width),

            image_width - 1

        )


        landmark_y = min(

            int(landmark.y * image_height),

            image_height - 1

        )


        landmark_point.append(
            [
                landmark_x,
                landmark_y
            ]
        )


    return landmark_point


# ============================================================
# PRE-PROCESS LANDMARK
# ============================================================

def pre_process_landmark(landmark_list):

    temp_landmark_list = copy.deepcopy(
        landmark_list
    )


    # --------------------------------------------------------
    # Convert to relative coordinates
    # --------------------------------------------------------

    base_x = 0

    base_y = 0


    for index, landmark_point in enumerate(
        temp_landmark_list
    ):

        if index == 0:

            base_x = landmark_point[0]

            base_y = landmark_point[1]


        temp_landmark_list[index][0] -= base_x

        temp_landmark_list[index][1] -= base_y


    # --------------------------------------------------------
    # Convert to 1D list
    # --------------------------------------------------------

    temp_landmark_list = list(

        itertools.chain.from_iterable(
            temp_landmark_list
        )

    )


    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------

    max_value = max(
        list(
            map(
                abs,
                temp_landmark_list
            )
        )
    )


    # Prevent division by zero

    if max_value == 0:

        max_value = 1


    def normalize_(n):

        return n / max_value


    temp_landmark_list = list(

        map(
            normalize_,
            temp_landmark_list
        )

    )


    return temp_landmark_list


# ============================================================
# LOGGING CSV
# ============================================================

def logging_csv(number, mode, landmark_list):

    # Recognition mode

    if mode == 0:

        return


    # Camera / dataset capture

    if (

        (mode == 1 or mode == 2)

        and

        (0 <= number <= 35)

    ):

        try:

            with open(

                CSV_PATH,

                "a",

                newline=""

            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        number,
                        *landmark_list
                    ]
                )

        except Exception as e:

            print(
                "CSV writing error:",
                e
            )


# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(image, landmark_point):

    if len(landmark_point) == 0:

        return image


    # MediaPipe hand connections

    connections = [

        # Thumb
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        # Index
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        # Middle
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        # Ring
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        # Little
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        # Palm
        (0, 17)

    ]


    # --------------------------------------------------------
    # Draw lines
    # --------------------------------------------------------

    for start, end in connections:

        cv.line(

            image,

            tuple(landmark_point[start]),

            tuple(landmark_point[end]),

            (0, 0, 0),

            6

        )


        cv.line(

            image,

            tuple(landmark_point[start]),

            tuple(landmark_point[end]),

            (255, 255, 255),

            2

        )


    # --------------------------------------------------------
    # Draw key points
    # --------------------------------------------------------

    for index, landmark in enumerate(
        landmark_point
    ):

        radius = 8 if index in [
            4,
            8,
            12,
            16,
            20
        ] else 5


        cv.circle(

            image,

            (
                landmark[0],
                landmark[1]
            ),

            radius,

            (255, 255, 255),

            -1

        )


        cv.circle(

            image,

            (
                landmark[0],
                landmark[1]
            ),

            radius,

            (0, 0, 0),

            1

        )


    return image


# ============================================================
# DRAW BOUNDING RECTANGLE
# ============================================================

def draw_bounding_rect(
    use_brect,
    image,
    brect
):

    if use_brect:

        cv.rectangle(

            image,

            (
                brect[0],
                brect[1]
            ),

            (
                brect[2],
                brect[3]
            ),

            (0, 0, 0),

            1

        )


    return image


# ============================================================
# DRAW HAND INFORMATION
# ============================================================

def draw_info_text(

    image,
    brect,
    handedness,
    hand_sign_text

):

    # Make sure rectangle is inside image

    x1 = max(
        0,
        brect[0]
    )

    y1 = max(
        22,
        brect[1]
    )

    x2 = max(
        x1 + 1,
        brect[2]
    )


    # Background

    cv.rectangle(

        image,

        (
            x1,
            y1
        ),

        (
            x2,
            y1 - 22
        ),

        (0, 0, 0),

        -1

    )


    # Handedness

    info_text = (
        handedness
        .classification[0]
        .label
    )


    if hand_sign_text != "":

        info_text = (

            info_text
            + ":"
            + hand_sign_text

        )


    cv.putText(

        image,

        info_text,

        (
            x1 + 5,
            y1 - 4
        ),

        cv.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        1,

        cv.LINE_AA

    )


    return image


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_info(

    image,
    fps,
    mode,
    number

):

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv.putText(

        image,

        "FPS:" + str(fps),

        (10, 30),

        cv.FONT_HERSHEY_SIMPLEX,

        1.0,

        (0, 0, 0),

        4,

        cv.LINE_AA

    )


    cv.putText(

        image,

        "FPS:" + str(fps),

        (10, 30),

        cv.FONT_HERSHEY_SIMPLEX,

        1.0,

        (255, 255, 255),

        2,

        cv.LINE_AA

    )


    # --------------------------------------------------------
    # Mode
    # --------------------------------------------------------

    mode_string = [

        "Recognition",

        "Logging Key Point",

        "Capturing Landmarks From Provided Dataset Mode"

    ]


    if 0 <= mode <= 2:

        cv.putText(

            image,

            "MODE:" + mode_string[mode],

            (10, 90),

            cv.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            1,

            cv.LINE_AA

        )


    # --------------------------------------------------------
    # Number
    # --------------------------------------------------------

    if 1 <= mode <= 2:

        if 0 <= number <= 35:

            cv.putText(

                image,

                "NUM:" + str(number),

                (10, 110),

                cv.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                1,

                cv.LINE_AA

            )


    return image


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":

    main()