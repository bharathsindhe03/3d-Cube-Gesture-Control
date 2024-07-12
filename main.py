import pygame
import numpy as np
import cv2
import mediapipe as mp

# Initialize pygame
pygame.init()

# Set up the display for the 3D cube
width, height = 800, 600
cube_screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('3D Cube Rotation')

# Define cube vertices
initial_vertices = np.array([
    [1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1],
    [-1, 1, -1],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1]
])

# Define cube faces
faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 3, 7, 4),
    (1, 2, 6, 5)
]

# Define face colors
colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255)   # Cyan
]

# Rotation matrices
def rotation_matrix_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_y(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_matrix_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Function to draw the cube on the screen
def draw_cube(screen, rotated_vertices):
    # Project 3D vertices to 2D
    projected_vertices = []
    for vertex in rotated_vertices:
        x, y, z = vertex
        f = 200 / (z + 5)
        x, y = x * f + width // 2, -y * f + height // 2
        projected_vertices.append((x, y))

    # Calculate face depths and sort faces by depth
    face_depths = []
    for i, face in enumerate(faces):
        depth = sum(rotated_vertices[vertex][2] for vertex in face) / 4
        face_depths.append((depth, i))
    face_depths.sort(reverse=True)

    # Draw faces
    for depth, i in face_depths:
        face = faces[i]
        pygame.draw.polygon(screen, colors[i], [projected_vertices[vertex] for vertex in face])

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define finger landmarks for status detection
def get_finger_status(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_tip = mp_hands.HandLandmark.THUMB_TIP
    thumb_ip = mp_hands.HandLandmark.THUMB_IP

    finger_status = [0, 0, 0, 0, 0]

    # Determine finger status based on landmark positions
    for i, tip in enumerate(finger_tips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_status[i + 1] = 1
    
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        finger_status[0] = 1

    return finger_status

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Main loop for hand detection and action handling
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    angle_x, angle_y, angle_z = 0, 0, 0
    scale = 1.0
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Read frame from the webcam
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for natural viewing
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get finger status and perform corresponding actions
                finger_status = get_finger_status(hand_landmarks)
                
                if finger_status == [0, 1, 0, 0, 0]:
                    angle_x += 0.1  # Rotate around X-axis
                elif finger_status == [0, 1, 1, 0, 0]:
                    angle_x -= 0.1
                elif finger_status == [0, 1, 1, 1, 0]:
                    angle_y += 0.1  # Rotate around Y-axis
                elif finger_status == [0, 1, 1, 1, 1]:
                    angle_y -= 0.1  # Rotate around Y-axis
                elif finger_status == [1, 0, 0, 0, 0]:
                    angle_z += 0.1  # Rotate around Z-axis
                elif finger_status == [1, 0, 0, 0, 1]:
                    angle_z -= 0.1  # Rotate around Z-axis
                elif finger_status == [1, 1, 1, 1, 1]:
                    scale += 0.1  # Increase scale
                elif finger_status == [0, 0, 0, 0, 0]:
                    scale -= 0.1  # Increase scale

        # Scale and rotate vertices
        scaled_vertices = initial_vertices * scale
        rotation_x = rotation_matrix_x(angle_x)
        rotation_y = rotation_matrix_y(angle_y)
        rotation_z = rotation_matrix_z(angle_z)
        rotated_vertices = np.dot(scaled_vertices, rotation_z)
        rotated_vertices = np.dot(rotated_vertices, rotation_y)
        rotated_vertices = np.dot(rotated_vertices, rotation_x)

        # Clear the cube screen
        cube_screen.fill((0, 0, 0))

        # Draw the cube in the cube window
        draw_cube(cube_screen, rotated_vertices)

        # Display the webcam feed in a separate window
        cv2.imshow('Webcam Feed', image)

        # Update the cube display
        pygame.display.flip()
        clock.tick(60)  # Control the frame rate

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources and close all windows
pygame.quit()
cap.release()
cv2.destroyAllWindows()
