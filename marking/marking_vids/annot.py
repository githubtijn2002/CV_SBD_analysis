def create_annot():
    annotation = {
        "info": {
            "description": "Manual COCO keypoint annotation",
            "version": "1.0"
        },
            "images": [],
            "annotations": [],
            "categories": [],
        }
    return annotation

def annotate_coco_keypoints(image_path, annotation, idx_file, save_to=None):
    """
    Annotate COCO keypoints on an image using matplotlib.
    Args:
        image_path (str): Path to the image.
        annotation (dict): COCO annotation dictionary.
        idx_file (str): Path to the index file to keep track of image IDs.
        save_to (str, optional): Path to save the annotations. If None, annotations are not saved.
    """
    import matplotlib
    matplotlib.use('TkAgg')

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import json
    import os

    COCO_KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    clicked_keypoints = []
    annotations = []  # Store (x, y) and text annotations for redrawing
    current_index = [0]

    img = mpimg.imread(image_path)
    image_height, image_width = img.shape[:2]

    def redraw_points():
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
        if current_index[0] < len(COCO_KEYPOINT_NAMES):
            ax.set_title(f"Click: {COCO_KEYPOINT_NAMES[current_index[0]]}")
        else:
            ax.set_title("Done! Closing the window in 1 seconds...")
            # escape the function and close the window after 2 seconds
            plt.pause(1)
            plt.close(fig)
            
        for i in range(len(annotations)):
            x, y = annotations[i]
            ax.plot(x, y, 'ro', markersize=4)
            ax.text(x, y, COCO_KEYPOINT_NAMES[i], color='white', fontsize=8)
        fig.canvas.draw()

    def onclick(event):
        if current_index[0] >= len(COCO_KEYPOINT_NAMES):
            return
        if event.xdata is not None and event.ydata is not None:
            x, y = float(event.xdata), float(event.ydata)
            clicked_keypoints.extend([x, y, 2])
            annotations.append((x, y))
            current_index[0] += 1
            redraw_points()

    def onkey(event):
        if event.key == 'ctrl+z' and current_index[0] > 0:
            # Remove last keypoint
            current_index[0] -= 1
            clicked_keypoints[-3:] = []  # Remove last x, y, vis
            annotations.pop()
            redraw_points()

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(f"Click: {COCO_KEYPOINT_NAMES[current_index[0]]}")
    ax.axis('off')

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    try:
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
    except Exception:
        try:
            manager.window.showMaximized()
        except Exception:
            pass

    plt.show(block=True)
    with open(idx_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'image_id' in line:
                image_id = int(line.split('=')[1].strip())
                break
        else:
            image_id = 1        
    # check if the image_id already exists in the annotations
    if image_id in [img['id'] for img in annotation['images']]:
        print(f"Image ID {image_id} already exists in the annotations.")
        return annotation
    
    annotation['images'].append({
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": image_width,
        "height": image_height
    })
    annotation['annotations'].append({
        "image_id": image_id,
        "category_id": 1,
        "keypoints": clicked_keypoints,
        "num_keypoints": len(clicked_keypoints) // 3,
        "area": 0,
        "bbox": [0, 0, 0, 0],
        "iscrowd": 0
    })
    annotation['categories'].append({
        "id": 1,
        "name": "person",
        "supercategory": "person",
        "keypoints": COCO_KEYPOINT_NAMES,
        "skeleton": []
    })

    with open(idx_file, 'w') as f:
        for line in lines:
            if 'image_id' in line:
                f.write(f'image_id={image_id + 1}\n')
            else:
                f.write(line)
    if save_to:
        with open(save_to, 'w') as f:
            json.dump(annotation, f, indent=4)
        print(f"Annotations saved to {save_to}")
        return
    else:
        print("Annotations not saved. Set save_to to a file path to save. Returning annotation dict.")
    return annotation

def annot_dir(image_dir, json_dir, idx_dir, annotation):
    """
    Annotate images in a directory and save the annotations to a JSON file.
    
    Args:
        image_dir (str): Path to the directory containing images.
        json_dir (str): Path to the directory where the JSON file will be saved.
        annotation (dict): COCO annotation dictionary.
    """
    import os
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    n_images = len(os.listdir(image_dir))
    for idx, image in enumerate(os.listdir(image_dir)):
        if not image.endswith('.jpg'):
            continue
        image_path = os.path.join(image_dir, image)
        if os.path.isfile(image_path):
            annotation = annotate_coco_keypoints(image_path=image_path, annotation=annotation, idx_file=idx_dir, save_to=None)
        if idx == n_images - 1:
            annotation = annotate_coco_keypoints(image_path, json=annotation, save_to=json_dir)
            print(f"Annotations saved to {os.path.join(json_dir.split('/')[1], 'annotations.json')}")
            break