print("Loading libraries...")
print(" - Loading lib: (time)...")
import time
print(" - (time) loaded successfully!")

lib_start_time = time.time()

print(" - Loading lib: (face_recognition)...")
import face_recognition
print(" - (face_recognition) loaded successfully!")

print(" - Loading lib: (numpy)...")
import numpy as np
print(" - (numpy) loaded successfully!")

print(" - Loading lib: (os)...")
import os
print(" - (os) loaded successfully!")
print()
print()
lib_end_time = time.time()
lib_elapsed_time = lib_end_time - lib_start_time

print("Libraries loaded successfully! :)")
print(f"Elapsed time: {lib_elapsed_time:.4f} seconds!")

print()
print()
print("Starting Model Training...")
model_start_time = time.time()

# Store the face encodings and names
known_face_encodings = []
known_face_names = []

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the face images are stored
# UNCOMMENT for Jetson!
# faces_directory = "/home/jetson/Face_Recognition_AIDL/test_images/"
# untrained_directory = "/home/jetson/Face_Recognition_AIDL/untrained_data/"

# For Windows


faces_directory = os.path.join(script_dir, "./trained_data")
untrained_directory = os.path.join(script_dir, "./untrained_data")

# Load the known faces and learn to recognize them
for filename in os.listdir(faces_directory):
    if filename.endswith(".png"):
        # Load each image file
        image = face_recognition.load_image_file(f"{faces_directory}/{filename}")
        # image = face_recognition.load_image_file(f"{faces_directory}{filename}") Jetson!

        # Get the face encodings for each face in each image file
        # Since there could be more than one face in each image, it returns a list of encodings.
        # Assume each image only has one face, take the first encoding only
        face_encoding = face_recognition.face_encodings(image)[0]

        # Add face encoding for current image with corresponding label (name) to the lists
        known_face_encodings.append(face_encoding)
        known_face_names.append(filename.split('.')[0]) # Remove the file extension from the filename
print("Model Successfully Trained! :)")

model_end_time = time.time()
model_elapsed_time = model_end_time - model_start_time
print(f"Elapsed time: {model_elapsed_time:.4f} seconds!")
print()
print()
print("Trained model with the following faces:")

for name in known_face_names:
    print(name)

while True:
    print("\nChoose an untrained image to recognize:")
    untrained_images = [filename.split('.')[0] for filename 
    in os.listdir(untrained_directory) 
    if filename.endswith(".png")]

    for idx, name in enumerate(untrained_images, start=1):
        print(f"{idx}. {name}")

    print("Enter 'end' to finish.")
    chosen_name = input("Choose an image: ")

    if chosen_name.lower() == 'end':
        break

    if chosen_name in untrained_images:
        recognition_start_time = time.time() # line 77
        # Perform face recognition, it uses the chosen image from the untrained_data folder
        # Uncomment for Jetson
        image_to_recognize = face_recognition.load_image_file(f"{untrained_directory}{chosen_name}.png") # assuming the images are pngs
        # Uncomment for Windows
        # image_to_recognize = face_recognition.load_image_file(f"{untrained_directory}/{chosen_name}.png")
        unknown_face_encodings = face_recognition.face_encodings(image_to_recognize)

        if len(unknown_face_encodings) == 0:
            print()
            print("Could not find any faces :(")
        else:
            for unknown_face_encoding in unknown_face_encodings:
                # See if this unknown face encoding matches any of the known people
                results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

                name = "Unknown"
                if True in results:
                    first_match_index = results.index(True)
                    name = known_face_names[first_match_index]
                print()
                print("The Jetson Nano wagers that you are: " + name)
                recognition_end_time = time.time()
                recognition_elapsed_time = recognition_end_time - recognition_start_time
                print()
                print(f"Elapsed time: {recognition_elapsed_time:.4f} seconds!")

    else:
        print("Image not recognized. You probably made a typo :) Please try again.")

