import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import os

path = os.path.join("saved", "checkpoint", "models", "scp_image_recognition", "0401_225219","model_best.pth")
def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        config['data_loader']['args']['aug_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, weights_only=False)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="all", type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)


"""


def classify_image(model, image_path, class_names, device='cpu'):

    # 1. Set model to evaluation mode
    model.eval()
    model.to(device)
    
    # 2. Load and preprocess the image
    input_image = Image.open(image_path).convert('RGB')
    
    # 3. Define transformations (must match training preprocessing)
    transform = transforms.Compose([
        transforms.Resize(256),            # Example - adjust to your model
        transforms.CenterCrop(224),        # Example - adjust to your model
        transforms.ToTensor(),
        transforms.Normalize(              # Example - use your model's normalization
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 4. Apply transformations and add batch dimension
    input_tensor = transform(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 5. Run inference
    with torch.no_grad():
        output = model(input_batch)
    
    # 6. Get predictions
    probabilities = F.softmax(output, dim=1)
    confidence, pred_class_idx = torch.max(probabilities, 1)
    
    # 7. Return results
    return class_names[pred_class_idx.item()], confidence.item()

# Example usage:
model = ...  # Your trained model
class_names = ['cat', 'dog', 'bird']  # Your class names
image_path = 'test_image.jpg'

pred_class, confidence = classify_image(model, image_path, class_names, device='cuda')
print(f'Predicted: {pred_class} with confidence: {confidence:.2f}')



"""