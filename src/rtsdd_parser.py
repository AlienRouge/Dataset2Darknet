import csv
from src.common_config import *
import json
from tqdm import tqdm

# TO CHANGE
RTSD_ROOT_PATH = r"C:\Users\rudov\Desktop\Datasets\RAW\RTSD"
JSON_PATH = r"C:\Users\rudov\Desktop\Datasets\RAW\RTSD\detect_lables_v0.8.json"

RESIZE_PERCENTAGE = 1.0  # 0.6
DB_PREFIX = 'rtsdd-'
USE_JSON = True

ANNOTATIONS_FILE_NAME = "full-gt.csv"
IMAGES_DIR_NAME = "rtsd-frames/"


def initialize_traffic_sign_classes():
    traffic_sign_classes.clear()
    print("Initialize classes..", end="")
    if USE_JSON:
        print("Using JSON values..")
        with open(JSON_PATH) as f:
            raw_json = json.load(f)
            for key, value in raw_json.items():
                traffic_sign_classes[key] = value
                print(key, value)
    init_classes()
    traffic_sign_classes[str(OTHER_CLASS) + "-" + OTHER_CLASS_NAME] = []


# It depends on the row format
def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = get_img_dim_plt(input_img)
    image_width = int(real_img_width * RESIZE_PERCENTAGE)
    image_height = int(real_img_height * RESIZE_PERCENTAGE)
    width_proportion = (real_img_width / image_width)
    height_proportion = (real_img_height / image_height)

    object_lb_x1 = float(row[1]) / width_proportion
    object_lb_y1 = float(row[2]) / height_proportion
    object_width = float(row[3]) / width_proportion
    object_height = float(row[4]) / height_proportion

    object_class = row[5]
    object_class_adjusted = adjust_object_class(object_class)  # Adjust class category

    if (SHOW_IMG):
        show_img(resize_img_plt(input_img, image_width, image_height), object_lb_x1, object_lb_y1, object_width,
                 object_height)

    return parse_darknet_format(object_class_adjusted, image_width, image_height,
                                object_lb_x1, object_lb_y1, object_lb_x1 + object_width, object_lb_y1 + object_height)


def update_global_variables(train_pct, test_pct, color_mode, verbose, false_data, output_img_ext):
    global TRAIN_PROB, TEST_PROB, COLOR_MODE, SHOW_IMG, ADD_FALSE_DATA, OUTPUT_IMG_EXTENSION
    TRAIN_PROB = train_pct
    TEST_PROB = test_pct
    COLOR_MODE = color_mode
    SHOW_IMG = verbose
    ADD_FALSE_DATA = false_data
    OUTPUT_IMG_EXTENSION = output_img_ext


# Function for reading the images
def read_dataset(output_train_text_path, output_test_text_path, output_train_dir_path, output_test_dir_path):
    img_labels = {}  # Set of images and its labels [filename]: [()]
    update_db_prefix(DB_PREFIX)
    initialize_traffic_sign_classes()
    initialize_classes_counter()

    train_text_file = open(output_train_text_path, "a+")
    test_text_file = open(output_test_text_path, "a+")

    annotations_file_path = RTSD_ROOT_PATH + "/" + ANNOTATIONS_FILE_NAME
    images_dir_path = RTSD_ROOT_PATH + "/" + IMAGES_DIR_NAME

    print("Reading dataset..", end="")
    if os.path.isfile(annotations_file_path) & os.path.isdir(images_dir_path):
        gt_file = open(annotations_file_path)  # Annotations file
        gt_reader = csv.reader(gt_file, delimiter=',')
        next(gt_reader)

        cntrf = 0
        cntrt = 0
        # WRITE ALL THE DATA IN A DICTIONARY (TO GROUP LABELS ON SAME IMG)
        for row in gt_reader:
            filename = row[0]
            file_path = images_dir_path + filename

            if os.path.isfile(file_path):
                input_img = read_img_plt(file_path)
                # print(row)
                darknet_label = calculate_darknet_format(input_img, row)
                # print(darknet_label)
                object_class_adjusted = int(darknet_label.split()[0])

                if filename not in img_labels.keys():  # If it is the first label for that img
                    img_labels[filename] = [file_path]

                # Add only useful labels (not false negatives)
                if object_class_adjusted != OTHER_CLASS:
                    img_labels[filename].append(darknet_label)
                    cntrt += 1
                else:
                    cntrf += 1
        gt_file.close()
    else:
        print("In folder " + RTSD_ROOT_PATH + " there are missing files. ")
    print(f"OK. Proceeded {len(img_labels)} images.")
    # COUNT FALSE NEGATIVES (IMG WITHOUT LABELS)
    total_false_negatives_dir = {}
    total_annotated_images_dir = {}
    # print(img_labels)
    for filename in img_labels.keys():
        img_label_subset = img_labels[filename]
        if len(img_label_subset) == 1:
            total_false_negatives_dir[filename] = img_label_subset
        else:
            total_annotated_images_dir[filename] = img_label_subset

    total_annotated_images = len(img_labels.keys()) - len(total_false_negatives_dir.keys())
    total_false_negatives = len(total_false_negatives_dir.keys())
    max_false_data = round(total_annotated_images * TRAIN_PROB)  # False data: False negative + background

    print("Total Labeled: " + str(total_annotated_images) + " == " + str(len(total_annotated_images_dir.keys())))
    print("Total Non-Labeled: " + str(total_false_negatives))

    # ADD FALSE IMAGES TO TRAIN
    if total_false_negatives > max_false_data:
        total_false_negatives = max_false_data

    if ADD_FALSE_DATA:
        add_false_negatives(total_false_negatives, total_false_negatives_dir, output_train_dir_path, train_text_file)

    #  max_imgs = 1000
    # for filename in total_annotated_images_dir.keys():
    for i in tqdm(range(len(total_annotated_images_dir.keys()))):
        _key = list(total_annotated_images_dir.keys())[i]
        input_img_file_path = img_labels[_key][0]
        # Read image from image_file_path
        input_img = read_img(input_img_file_path)
        input_img = resize_img_percentage(input_img, RESIZE_PERCENTAGE)  # Resize img
        input_img_labels = img_labels[_key][1:]

        # Get percentage for train and another for testing
        train_file = rand.choices([True, False], [TRAIN_PROB, TEST_PROB])[0]
        output_filename = DB_PREFIX + _key[:-4]

        if train_file:
            write_data(output_filename, input_img, input_img_labels,
                       train_text_file, output_train_dir_path, train_file)
        else:
            write_data(output_filename, input_img, input_img_labels,
                       test_text_file, output_test_dir_path, train_file)

    train_text_file.close()
    test_text_file.close()

    return classes_counter_train, classes_counter_test
