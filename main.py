import pygame
import numpy as np
import cv2
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
cube_screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('3D Cube Rotation')

# Define the initial vertices of the cube
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

# Define the faces of the cube using the vertices indices
faces = [
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (2, 3, 7, 6),
    (0, 3, 7, 4),
    (1, 2, 6, 5)
]

# Colors for each face of the cube
colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255)  # Cyan
]

# Rotation matrices for X, Y, and Z axes
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

# Function to draw the cube
def draw_cube(screen, rotated_vertices):
    projected_vertices = []
    for vertex in rotated_vertices:
        x, y, z = vertex
        f = 200 / (z + 5)  # Perspective projection factor
        x, y = x * f + width // 2, -y * f + height // 2
        projected_vertices.append((x, y))

    # Sort faces based on depth to handle occlusion
    face_depths = []
    for i, face in enumerate(faces):
        depth = sum(rotated_vertices[vertex][2] for vertex in face) / 4
        face_depths.append((depth, i))
    face_depths.sort(reverse=True)

    # Draw the faces of the cube
    for depth, i in face_depths:
        face = faces[i]
        pygame.draw.polygon(screen, colors[i], [projected_vertices[vertex] for vertex in face])

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to get the status of fingers (extended or not)
def get_finger_status(hand_landmarks):
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                   mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP,
                   mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_tip = mp_hands.HandLandmark.THUMB_TIP
    thumb_ip = mp_hands.HandLandmark.THUMB_IP

    finger_status = [0, 0, 0, 0, 0]  # Thumb, Index, Middle, Ring, Pinky

    # Check if each finger is extended
    for i, tip in enumerate(finger_tips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            finger_status[i + 1] = 1
    
    # Check if thumb is extended
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        finger_status[0] = 1

    return finger_status

# Open the webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    # Initialize angles and scale for the cube
    angle_x, angle_y, angle_z = 0, 0, 0
    scale = 1.0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Process the image with MediaPipe Hands
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
             
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                finger_status = get_finger_status(hand_landmarks)
                
                # Adjust angles and scale based on finger status
                if finger_status == [0, 1, 0, 0, 0]:
                    angle_x += 0.1  
                elif finger_status == [0, 1, 1, 0, 0]:
                    angle_x -= 0.1  
                elif finger_status == [0, 1, 1, 1, 0]:
                    angle_y += 0.1  
                elif finger_status == [0, 1, 1, 1, 1]:
                    angle_y -= 0.1  
                elif finger_status == [1, 0, 0, 0, 0]:
                    angle_z += 0.1  
                elif finger_status == [1, 0, 0, 0, 1]:
                    angle_z -= 0.1  
                elif finger_status == [1, 1, 1, 1, 1]:
                    scale += 0.1  
                elif finger_status == [0, 0, 0, 0, 0]:
                    scale -= 0.1  
                    if scale <= 0.1:
                        scale = 0.1

        # Apply transformations to the vertices of the cube
        scaled_vertices = initial_vertices * scale
        rotation_x = rotation_matrix_x(angle_x)
        rotation_y = rotation_matrix_y(angle_y)
        rotation_z = rotation_matrix_z(angle_z)
        rotated_vertices = np.dot(scaled_vertices, rotation_z)
        rotated_vertices = np.dot(rotated_vertices, rotation_y)
        rotated_vertices = np.dot(rotated_vertices, rotation_x)

        # Clear the screen
        cube_screen.fill((0, 0, 0))

        # Draw the transformed cube
        draw_cube(cube_screen, rotated_vertices)

        # Show the webcam feed with hand landmarks
        cv2.imshow('Webcam', image)
        
        # Update the Pygame display
        pygame.display.flip()
        clock.tick(60)  
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

pygame.quit()
cap.release()
cv2.destroyAllWindows()
