import os
import random
import shutil

# --- Configuration ---

# The root directory containing all the person subfolders.
# Based on your path: D:\...\lfw-deepfunneled\lfw-deepfunneled\
SOURCE_ROOT = r"../../../../casia-webface"

# The directory where the new 'train' and 'test' folders will be created.
DESTINATION_ROOT = r"../../data/humans"

# Split configuration
MIN_IMAGES_REQUIRED = 20
TRAIN_COUNT = 20
TEST_COUNT = 0
NUM_PEOPLE_TO_SELECT = 1000

def split_and_copy_images():
    """
    1. Finds all people folders in the SOURCE_ROOT.
    2. Filters for people with at least MIN_IMAGES_REQUIRED.
    3. Selects NUM_PEOPLE_TO_SELECT random people.
    4. Splits their images into TRAIN_COUNT and TEST_COUNT and copies them
       to the respective DESTINATION_ROOT subfolders.
    """
    print(f"Starting dataset split process. Source: {SOURCE_ROOT}")

    # 1. Check if source directory exists
    if not os.path.isdir(SOURCE_ROOT):
        print(f"Error: Source directory not found at '{SOURCE_ROOT}'")
        return

    # Prepare destination directories
    train_dir = os.path.join(DESTINATION_ROOT, 'train')
    test_dir = os.path.join(DESTINATION_ROOT, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(f"Destination directories ensured: {train_dir} and {test_dir}")

    # 2. List all person folders
    all_people_folders = [
        name for name in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, name))
    ]

    # 3. Filter people based on image count
    eligible_people = {}
    print("Checking image counts for all people...")
    for person_name in all_people_folders:
        person_path = os.path.join(SOURCE_ROOT, person_name)
        # List files in the person's folder, filtering for common image extensions
        try:
            images = [
                f for f in os.listdir(person_path)
                if os.path.isfile(os.path.join(person_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
            ]

            if len(images) >= MIN_IMAGES_REQUIRED:
                eligible_people[person_name] = images
                # print(f" - {person_name}: {len(images)} images (Eligible)")
            else:
                print(f" - {person_name}: {len(images)} images (Skipped - Needs >= {MIN_IMAGES_REQUIRED})")
        except Exception as e:
            print(f"Could not process folder {person_name}: {e}")

    eligible_names = list(eligible_people.keys())
    print(f"\nFound {len(eligible_names)} people eligible for selection.")

    # 4. Select random people
    if len(eligible_names) < NUM_PEOPLE_TO_SELECT:
        print(f"Error: Only {len(eligible_names)} eligible people found, but need {NUM_PEOPLE_TO_SELECT}.")
        return

    selected_people_names = random.sample(eligible_names, NUM_PEOPLE_TO_SELECT)
    print(f"\nSelected {NUM_PEOPLE_TO_SELECT} random people: {selected_people_names}")

    # 5. Process and copy images for each selected person
    for person_name in selected_people_names:
        print(f"\nProcessing {person_name}...")

        all_images = eligible_people[person_name]
        
        # Ensure we have enough files to take the required train and test counts
        if len(all_images) < TRAIN_COUNT + TEST_COUNT:
            print(f"Warning: {person_name} has only {len(all_images)} images. Need {TRAIN_COUNT + TEST_COUNT}. Skipping split.")
            continue

        # Randomly shuffle the list of images to ensure randomness
        random.shuffle(all_images)

        # Split into train and test sets
        train_files = all_images[:TRAIN_COUNT]
        test_files = all_images[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

        # 6. Copy files to train directory
        dest_train_path = os.path.join(train_dir, person_name)
        os.makedirs(dest_train_path, exist_ok=True)

        for file_name in train_files:
            src = os.path.join(SOURCE_ROOT, person_name, file_name)
            dest = os.path.join(dest_train_path, file_name)
            shutil.copy2(src, dest)

        print(f" - Copied {len(train_files)} images to training folder: {dest_train_path}")

        # 7. Copy files to test directory
        dest_test_path = os.path.join(test_dir, person_name)
        os.makedirs(dest_test_path, exist_ok=True)

        for file_name in test_files:
            src = os.path.join(SOURCE_ROOT, person_name, file_name)
            dest = os.path.join(dest_test_path, file_name)
            shutil.copy2(src, dest)

        print(f" - Copied {len(test_files)} images to testing folder: {dest_test_path}")


if __name__ == "__main__":
    split_and_copy_images()
