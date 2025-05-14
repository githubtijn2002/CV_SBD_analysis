def random_sample(video_path, output_path, n_samples=5, spread='rand'):
    """
    Randomly samples frames from a video file.
    :param video_path: Path to the video file.
    :param n_samples: Number of frames to sample.
    :param spread: Method of sampling ('rand' for random, 'even' for evenly spaced).
    :return: List of sampled frames.
    """
    import cv2
    import random
    import os
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_samples > total_frames:
        raise ValueError("Number of samples exceeds total frames in the video.")
    if spread == 'rand':
        frame_indices = sorted(random.sample(range(total_frames), n_samples))
    elif spread == 'even':
        frame_indices = [i * (total_frames // n_samples) for i in range(n_samples)]
    else:
        raise ValueError("Invalid spread method. Use 'rand' or 'even'.")
    sampled_frames = []
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)

    cap.release()
    # save the sampled frames as .jpg files in the output path
    video_name = os.path.basename(video_path).split('.')[0]
    for i, frame in enumerate(sampled_frames):
        cv2.imwrite(os.path.join(output_path, f"{video_name}_{i+1}.jpg"), frame)
        
    print(f"Sampled {n_samples} frames from {video_path} and saved to {output_path}.")

def clear_images(video_name, image_dir):
    """
    Clears images from the image directory.
    :param video_name: Name of the video file.
    :param image_dir: Path to the image directory.
    """
    import os
    count = 0
    for file in os.listdir(image_dir):
        if file.startswith(video_name):
            os.remove(os.path.join(image_dir, file))
            count += 1
    if count == 0:
        print(f"No images found for video {video_name} in {image_dir}.")
    else:
        print(f"Removed {count} images from {image_dir} for video {video_name}.")

def vid_loop(video_dir, image_dir, n_samples=10, spread='even'):
    """
    Loops through all videos in a directory and samples frames from each video.
    :param video_dir: Path to the directory containing video files.
    :param image_dir: Path to the directory where images will be saved.
    :param n_samples: Number of frames to sample from each video.
    :param spread: Method of sampling ('rand' for random, 'even' for evenly spaced).
    """    
    import os
    for video in os.listdir(video_dir):
        if not video.endswith(('.mp4', '.avi', '.mov')):
            print(f"Skipping non-video file: {video}")
            continue
        video_path = os.path.join(video_dir, video)

        # check if the output path exists, if not create it
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        # call the random_sample function
        random_sample(video_path, image_dir, n_samples, spread)
