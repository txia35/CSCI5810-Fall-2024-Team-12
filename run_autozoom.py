import argparse
import cv2
import numpy as np
import PIL
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from inference import Converter, HSVClassifier, InertiaClassifier, YoloV5
from inference.filters import filters
from run_utils import (
    get_ball_detections,
    get_main_ball,
    get_player_detections,
    update_motion_estimator,
)
from soccer import Match, Player, Team
from soccer.draw import AbsolutePath
from soccer.pass_event import Pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "--video",
    default="videos/soccer_possession.mp4",
    type=str,
    help="Path to the input video",
)
parser.add_argument(
    "--model", default="models/ball.pt", type=str, help="Path to the model"
)
parser.add_argument(
    "--passes",
    action="store_true",
    help="Enable pass detection",
)
parser.add_argument(
    "--possession",
    action="store_true",
    help="Enable possession counter",
)
args = parser.parse_args()

video = Video(input_path=args.video)
fps = video.video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
crop_width, crop_height = int(frame_width * 0.6), int(frame_height * 0.6)

# Object Detectors
player_detector = YoloV5()
ball_detector = YoloV5(model_path=args.model)

# HSV Classifier
hsv_classifier = HSVClassifier(filters=filters)
classifier = InertiaClassifier(classifier=hsv_classifier, inertia=20)

# Teams and Match
chelsea = Team(name="Chelsea", abbreviation="NAP", color=(230, 130, 180))
man_city = Team(name="Man City", abbreviation="BAR", color=(240, 230, 188))
teams = [chelsea, man_city]
match = Match(home=chelsea, away=man_city, fps=fps)
match.team_possession = man_city

# Trackers and motion estimator
player_tracker = Tracker(
    distance_function=mean_euclidean, distance_threshold=250, initialization_delay=3, hit_counter_max=90
)
ball_tracker = Tracker(
    distance_function=mean_euclidean, distance_threshold=150, initialization_delay=20, hit_counter_max=2000
)
motion_estimator = MotionEstimator()

# Paths and backgrounds
path = AbsolutePath()
possession_background = match.get_possession_background()
passes_background = match.get_passes_background()

# Initial parameters for controlling smooth movement
target_center_x, target_center_y = frame_width // 2, frame_height // 2
current_center_x, current_center_y = frame_width // 2, frame_height // 2
speed_x, speed_y = 0, 0
max_speed = 10  # Maximum speed for camera movement
acceleration = 1  # Acceleration factor for camera movement
deceleration_threshold = 50  # Distance at which deceleration starts
# New dimensions for pre-cropping to 90% of the original size
pre_crop_width = int(frame_width * 0.9)
pre_crop_height = int(frame_height * 0.9)
crop_ratio = 0.6

# Calculate start and end points to center the pre-crop
pre_crop_start_x = (frame_width - pre_crop_width) // 2
pre_crop_start_y = (frame_height - pre_crop_height) // 2
pre_crop_end_x = pre_crop_start_x + pre_crop_width
pre_crop_end_y = pre_crop_start_y + pre_crop_height

for i, frame in enumerate(video):
    # Pre-crop the frame to 90% of the original size to reduce misclassified audience members
    frame = frame[pre_crop_start_y:pre_crop_end_y, pre_crop_start_x:pre_crop_end_x]

    # Update dimensions for the cropped frame
    frame_height, frame_width = frame.shape[:2]
    crop_width, crop_height = int(frame_width * crop_ratio), int(frame_height * crop_ratio)

    # Continue with the rest of the processing using the pre-cropped frame
    players_detections = get_player_detections(player_detector, frame)
    ball_detections = get_ball_detections(ball_detector, frame)
    
    # Update trackers
    coord_transformations = update_motion_estimator(
        motion_estimator=motion_estimator,
        detections=players_detections + ball_detections,
        frame=frame,
    )
    player_track_objects = player_tracker.update(
        detections=players_detections, coord_transformations=coord_transformations
    )
    ball_track_objects = ball_tracker.update(
        detections=ball_detections, coord_transformations=coord_transformations
    )
    player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
    ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
    
    # Apply HSV and inertia classification to players
    player_detections = classifier.predict_from_detections(detections=player_detections, img=frame)
    players = Player.from_detections(detections=players_detections, teams=teams)
    match.update(players, get_main_ball(ball_detections))

    # Draw for possession if enabled, before final crop
    frame = PIL.Image.fromarray(frame)
    if args.possession:
        frame = Player.draw_players(players=players, frame=frame, confidence=False, id=True)
        frame = path.draw(
            img=frame,
            detection=get_main_ball(ball_detections).detection if ball_detections else None,
            coord_transformations=coord_transformations,
            color=match.team_possession.color,
        )
        # frame = match.draw_possession_counter(frame, counter_background=possession_background, debug=False)
    
    # Convert back to numpy array after drawing
    frame = np.array(frame)

    # print([player.team for player in players])

    # Filter players based on team validity
    valid_team_players = [player for player in players if player.team is not None]
    if len([p for p in valid_team_players if p is not None]) == 0:
        valid_team_players = players
    # players_to_consider = valid_team_players if valid_team_players else players
        
    min_x = 0
    max_x = frame_width
    min_y = 0
    max_y = frame_height

    # Calculate bounding box to include maximum valid players
    if len(valid_team_players) > 0:
        min_x = min(player.detection.points[0][0] for player in valid_team_players)
        max_x = max(player.detection.points[1][0] for player in valid_team_players)
        min_y = min(player.detection.points[0][1] for player in valid_team_players)
        max_y = max(player.detection.points[1][1] for player in valid_team_players)
    
    # Target center point for the zoomed area
    target_center_x = (min_x + max_x) // 2
    target_center_y = (min_y + max_y) // 2
    
    # Smooth camera movement with acceleration and deceleration
    delta_x = target_center_x - current_center_x
    delta_y = target_center_y - current_center_y
    
    # Update speed with acceleration or deceleration
    if abs(delta_x) > deceleration_threshold:
        speed_x = min(max_speed, speed_x + acceleration) if delta_x > 0 else max(-max_speed, speed_x - acceleration)
    else:
        speed_x = max(0, abs(speed_x) - acceleration) * (1 if speed_x > 0 else -1)
        
    if abs(delta_y) > deceleration_threshold:
        speed_y = min(max_speed, speed_y + acceleration) if delta_y > 0 else max(-max_speed, speed_y - acceleration)
    else:
        speed_y = max(0, abs(speed_y) - acceleration) * (1 if speed_y > 0 else -1)
    
    # Apply speed to current center
    current_center_x += speed_x
    current_center_y += speed_y

    # Calculate crop boundaries based on the smoothed current center
    start_x = max(0, current_center_x - crop_width // 2)
    start_y = max(0, current_center_y - crop_height // 2)
    end_x = min(frame_width, start_x + crop_width)
    end_y = min(frame_height, start_y + crop_height)
    
    # Crop frame
    # print(frame.size)
    frame = frame[start_y:end_y, start_x:end_x]
    frame = PIL.Image.fromarray(frame)
    # print(frame.size)
    frame = match.draw_possession_counter(frame, counter_background=possession_background, debug=False)
    frame = np.array(frame)
    # Write cropped frame to video
    video.write(frame)
