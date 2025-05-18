import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_transformers.data_transform import image_transforms
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import os
from PIL import Image



path = os.path.join("saved", "models", "scp_image_recognition", "0517_200516","model_best.pth")
image_path = os.path.join("test_data","image.jpeg")

def main(config):
    logger = config.get_logger('test')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data_loader = config.init_obj('data_loader', module_data)
    logger.info("[INFO] Executing data_loader")
    valid_data_loader = data_loader.split_validation()
    class_names = valid_data_loader.dataset.class_to_idx
    class_names = {v: k for k, v in class_names.items()}
    logger.info("[INFO] Class names : {} ".format(class_names))

    def load_model():
        model = config.init_obj('arch', module_arch)

        # get function handles of loss and metrics
        loss_fn = getattr(module_loss, config['loss'])
        metric_fns = [getattr(module_metric, met) for met in config['metrics']]
        logger.info('Loading checkpoint: {} ...'.format(config.resume))

        checkpoint = torch.load(config.resume, weights_only=False)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        logger.info('[INFO] Loaded the checkpoint: {} ...'.format(config.resume))
        return model


    # Loading the Model Architecture and Loading the Checkpoints from the Saved Model.
    model = load_model()
    logger.info("[INFO]- Successfully Loaded the Trained model")
    logger.info(model)

    # Importing the data
    logger.info("[INFO]- Importing the data from the validation set location for Testing")
    try:
        image = Image.open(image_path).convert('RGB')  # Convert to RGB (3 channels)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
    logger.info("[INFO]- Imported the image.jpeg file from the validation set location for Testing")

    # Transforming the image
    transform = image_transforms.get("test")
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # Preparing the Model for testing
    logger.info("[INFO] - Preparing the model for validation testing")
    model = model.to(device)
    model.eval() # Prepares the model in evaluation mode.


    with torch.no_grad():
        output = model(input_batch)
    print("model.output:", output.shape)
    probabilities = F.softmax(output, dim=1)
    confidence, pred_class_idx = torch.max(probabilities, 1)
    print("classnames: ", class_names)
    print("Predicted class index:", pred_class_idx.item())

    output_class = class_names[pred_class_idx.item()], confidence.item()


    return output_class


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="all", type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    output = main(config)