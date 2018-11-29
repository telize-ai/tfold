# tfold - Tensorflow Object Detector

## Introduction

This simple wrapper aims to provide a more programmatic alternative to the training pipeline provided by Tensorflow object detection research project. 

## Installation

    pip install git+https://github.com/telize-ai/tfold.git
    
## Usage

### Detection

    # Load the inference graph
    detector = Detector(
        graph_path='frozen_inference_graph.pb',
        label_path='label_map.config'
    )  
    
    # Run the detection
    detections = detector.detect_in_file('image.jpg')

    # Draw an image with bounding boxes
    detector.draw_vis(threshold=0.4)
    
### Training
    
    json_data = open(os.path.join(data_dir, 'samples.json')).read()
    samples = json.loads(json_data)

    train = Trainer(
        model_name='sport_brands',
        model_dir='tmp/sport_brands',
        model_type='faster'
    )

    for sample in samples['train']:
        filename = os.path.join(data_dir, sample['filename'])
        train.add_train_sample(filename, sample['boxes'])

    for sample in samples['eval']:
        filename = os.path.join(data_dir, sample['filename'])
        train.add_eval_sample(filename, sample['boxes'])

    train.start()
    

The `samples.json` file contains a json with the following format:

    samples = [
        {
            filename: str,
            boxes: [
                {
                    xmin: int,
                    xmax: int,
                    ymin: int,
                    ymax: int,
                    class_text: str,
                    class_int: int
                },
                ...
            ]
        },
        ...
    ]
