'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - save_checkpoint: save current best model (based on test metric).
    - save_onnx_model: save final model in onnx format.
'''
import os
import torch

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def save_checkpoint(epoch, model, test_metric):
    state = {
        'net': model.state_dict(),
        'test_metric': test_metric,
        'epoch': epoch,
    }

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.pth')

def save_onnx_model(model, dummy_input, verbose=False):
    input_names = [ "actual_input_1" ] + [ name for name, param in model.named_parameters() ]
    output_names = [ "output1" ]

    torch.onnx.export(model, dummy_input, "./model.onnx", verbose=verbose, 
                      input_names=input_names, output_names=output_names)