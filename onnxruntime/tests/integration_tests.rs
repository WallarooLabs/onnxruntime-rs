use std::{
    fs,
    io::{self, BufRead, BufReader},
    path::Path,
    time::Duration,
};

use onnxruntime::error::OrtDownloadError;

mod download {
    use super::*;

    use image::{imageops::FilterType, ImageBuffer, Luma, Pixel, Rgb};
    use ndarray::s;
    use test_log::test;

    use onnxruntime::{
        download::vision::{
            DomainBasedImageClassification, ImageClassification, ObjectDetectionImageSegmentation,
        },
        environment::Environment,
        tensor::{ndarray_tensor::NdArrayTensor, DynOrtTensor},
        GraphOptimizationLevel, LoggingLevel,
    };

    #[test]
    fn squeezenet_mushroom() {
        const IMAGE_TO_LOAD: &str = "mushroom.png";

        let environment = Environment::builder()
            .with_name("integration_test")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        let mut session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_downloaded(ImageClassification::SqueezeNet)
            .expect("Could not download model from file");

        let class_labels = get_imagenet_labels().unwrap();

        let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
        let output0_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(|d| d.unwrap())
            .collect();

        assert_eq!(input0_shape, [1, 3, 224, 224]);
        assert_eq!(output0_shape, [1, 1000]);

        // Load image and resize to model's shape, converting to RGB format
        let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("data")
                .join(IMAGE_TO_LOAD),
        )
        .unwrap()
        .resize(
            input0_shape[2] as u32,
            input0_shape[3] as u32,
            FilterType::Nearest,
        )
        .to_rgb8();

        // Python:
        // # image[y, x, RGB]
        // # x==0 --> left
        // # y==0 --> top

        // See https://github.com/onnx/models/blob/master/vision/classification/imagenet_inference.ipynb
        // for pre-processing image.
        // WARNING: Note order of declaration of arguments: (_,c,j,i)
        let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        });

        // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];
        for c in 0usize..3 {
            let mut channel_array = array.slice_mut(s![0usize, c, .., ..]);
            channel_array -= mean[c];
            channel_array /= std[c];
        }

        // Batch of 1
        let input_tensor_values = vec![array];

        // Perform the inference
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(input_tensor_values).unwrap();

        // Downloaded model does not have a softmax as final layer; call softmax on second axis
        // and iterate on resulting probabilities, creating an index to later access labels.
        let output: &DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>> = &outputs[0];
        let mut probabilities: Vec<(usize, f32)> = output
            .try_extract()
            .unwrap()
            .view()
            .softmax(ndarray::Axis(1))
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        // Sort probabilities so highest is at beginning of vector.
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        assert_eq!(
            class_labels[probabilities[0].0], "n07734744 mushroom",
            "Expecting class for {} to be a mushroom",
            IMAGE_TO_LOAD
        );

        assert_eq!(
            probabilities[0].0, 947,
            "Expecting class for {} to be a mushroom (index 947 in labels file)",
            IMAGE_TO_LOAD
        );

        // for i in 0..5 {
        //     println!(
        //         "class={} ({}); probability={}",
        //         labels[probabilities[i].0], probabilities[i].0, probabilities[i].1
        //     );
        // }
    }

    #[test]
    fn mnist_5() {
        const IMAGE_TO_LOAD: &str = "mnist_5.jpg";

        let environment = Environment::builder()
            .with_name("integration_test")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        let mut session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_downloaded(DomainBasedImageClassification::Mnist)
            .expect("Could not download model from file");

        let input0_shape: Vec<usize> = session.inputs[0].dimensions().map(|d| d.unwrap()).collect();
        let output0_shape: Vec<usize> = session.outputs[0]
            .dimensions()
            .map(|d| d.unwrap())
            .collect();

        assert_eq!(input0_shape, [1, 1, 28, 28]);
        assert_eq!(output0_shape, [1, 10]);

        // Load image and resize to model's shape, converting to RGB format
        let image_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = image::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("data")
                .join(IMAGE_TO_LOAD),
        )
        .unwrap()
        .resize(
            input0_shape[2] as u32,
            input0_shape[3] as u32,
            FilterType::Nearest,
        )
        .to_luma8();

        let array = ndarray::Array::from_shape_fn((1, 1, 28, 28), |(_, c, j, i)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        });

        // Batch of 1
        let input_tensor_values = vec![array];

        // Perform the inference
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(input_tensor_values).unwrap();

        let output: &DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>> = &outputs[0];
        let mut probabilities: Vec<(usize, f32)> = output
            .try_extract()
            .unwrap()
            .view()
            .softmax(ndarray::Axis(1))
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();

        // Sort probabilities so highest is at beginning of vector.
        probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        assert_eq!(
            probabilities[0].0, 5,
            "Expecting class for {} is '5' (not {})",
            IMAGE_TO_LOAD, probabilities[0].0
        );
    }

    // This test verifies that dynamically sized inputs and outputs work. It loads and runs
    // upsample.onnx, which was produced via:
    //
    // ```
    // import subprocess
    // from tensorflow import keras
    //
    // m = keras.Sequential([
    //     keras.layers.UpSampling2D(size=2)
    // ])
    // m.build(input_shape=(None, None, None, 3))
    // m.summary()
    // m.save('saved_model')
    //
    // subprocess.check_call([
    //     'python', '-m', 'tf2onnx.convert',
    //     '--saved-model', 'saved_model',
    //     '--opset', '12',
    //     '--output', 'upsample.onnx',
    // ])
    // ```
    #[test]
    fn upsample() {
        const IMAGE_TO_LOAD: &str = "mushroom.png";

        let environment = Environment::builder()
            .with_name("integration_test")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        let mut session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_from_file(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests")
                    .join("data")
                    .join("upsample.onnx"),
            )
            .expect("Could not open model from file");

        assert_eq!(
            session.inputs[0].dimensions().collect::<Vec<_>>(),
            [None, None, None, Some(3)]
        );
        assert_eq!(
            session.outputs[0].dimensions().collect::<Vec<_>>(),
            [None, None, None, Some(3)]
        );

        // Load image, converting to RGB format
        let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("data")
                .join(IMAGE_TO_LOAD),
        )
        .unwrap()
        .to_rgb8();

        let array = ndarray::Array::from_shape_fn((1, 224, 224, 3), |(_, j, i, c)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        });

        // Just one input
        let input_tensor_values = vec![array];

        // Perform the inference
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(input_tensor_values).unwrap();

        assert_eq!(outputs.len(), 1);
        let output = &outputs[0];

        // The image should have doubled in size
        assert_eq!(
            output.try_extract::<f32>().unwrap().view().shape(),
            [1, 448, 448, 3]
        );
    }

    #[test]
    fn frcnn_zero_object_detections() {
        const IMAGE_TO_LOAD: &str = "water.jpeg";

        let environment = Environment::builder()
            .with_name("integration_test")
            .with_log_level(LoggingLevel::Warning)
            .build()
            .unwrap();

        let mut session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_number_threads(1)
            .unwrap()
            .with_model_downloaded(ObjectDetectionImageSegmentation::FasterRcnn)
            .expect("Could not download model file");

        let input0_shape: Vec<Option<usize>> = session.inputs[0].dimensions().collect();
        let output0_shape: Vec<Option<usize>> = session.outputs[0].dimensions().collect();
        let output1_shape: Vec<Option<usize>> = session.outputs[1].dimensions().collect();
        let output2_shape: Vec<Option<usize>> = session.outputs[2].dimensions().collect();

        assert_eq!(input0_shape, [Some(3), None, None]);
        assert_eq!(output0_shape, [None, Some(4)]);
        assert_eq!(output1_shape, [None]);
        assert_eq!(output2_shape, [None]);

        let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests")
                .join("data")
                .join(IMAGE_TO_LOAD),
        )
        .unwrap()
        .resize(640, 480, FilterType::Nearest)
        .to_rgb8();

        let array = ndarray::Array::from_shape_fn((3, 480, 640), |(c, j, i)| {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        });

        // // Perform the inference
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            session.run(vec![array]).unwrap();

        assert_eq!(outputs.len(), 3);

        let boxes = &outputs[0];
        let labels = &outputs[1];
        let scores = &outputs[2];

        // There should be 0 bounding boxes
        assert_eq!(boxes.try_extract::<f32>().unwrap().view().shape(), [0, 4]);

        // There should be 0 labels
        assert_eq!(labels.try_extract::<i64>().unwrap().view().shape(), [0]);

        // There should be 0 scores
        assert_eq!(scores.try_extract::<f32>().unwrap().view().shape(), [0]);
    }
}

fn get_imagenet_labels() -> Result<Vec<String>, OrtDownloadError> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("synset.txt");
    if !labels_path.exists() {
        let url = "https://s3.amazonaws.com/onnx-model-zoo/synset.txt";
        println!("Downloading {:?} to {:?}...", url, labels_path);
        let resp = ureq::get(url)
            .timeout(Duration::from_secs(180)) // 3 minutes
            .call()
            .map_err(Box::new)
            .map_err(OrtDownloadError::UreqError)?;

        assert!(resp.has("Content-Length"));
        let len = resp
            .header("Content-Length")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap();
        println!("Downloading {} bytes...", len);

        let mut reader = resp.into_reader();

        let f = fs::File::create(&labels_path).unwrap();
        let mut writer = io::BufWriter::new(f);

        let bytes_io_count = io::copy(&mut reader, &mut writer).unwrap();

        assert_eq!(bytes_io_count, len as u64);
    }
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines()
        .map(|line| line.map_err(|io_err| OrtDownloadError::IoError(io_err)))
        .collect()
}
