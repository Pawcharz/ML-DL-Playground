import torch
from datetime import datetime

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
sum_writer = SummaryWriter('runs/chest_trainer_{}'.format(timestamp))

def train_one_epoch(model, training_loader, optimizer, loss_fn, accuracy_metric, cuda_device, epoch_index, logging_frequency):
  running_loss = 0.
  running_accuracy = 0.
  last_loss = 0.

  # Here, we use enumerate(training_loader) instead of
  # iter(training_loader) so that we can track the batch
  # index and do some intra-epoch reporting
  for i, data in enumerate(training_loader):
  
    # Every data instance is an input + label pair
    inputs = data['image'].to(cuda_device)
    labels = data['label'].to(cuda_device)
  
    # Zero your gradients for every batch!
    optimizer.zero_grad()

    # Make predictions for this batch
    outputs = model(inputs)

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    loss.backward()
    
    training_accuracy = accuracy_metric(outputs, labels)

    # Adjust learning weights
    optimizer.step()

    # Gather data and report
    running_loss += loss.item()
    running_accuracy += training_accuracy
    
    # print('batch {}', i)
    
    if (i+1) % logging_frequency == 0:
      last_loss = running_loss / logging_frequency # loss per batch
      last_accuracy = running_accuracy / logging_frequency # accuracy per batch
      print('  batch {} loss: {} training_accuracy: {}'.format(i + 1, last_loss, last_accuracy))
      tb_x = epoch_index * len(training_loader) + i + 1
      sum_writer.add_scalar('Loss/train', last_loss, tb_x)
      running_loss = 0.
      running_accuracy = 0.
  
  return last_loss


def train_many_epochs(epochs, model, training_loader, validation_loader, optimizer, loss_fn, accuracy_metric, cuda_device, epoch_index, logging_frequency):
  best_vloss = 1_000_000.
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  for epoch_number in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss =  train_one_epoch(
      model=model,
      training_loader=training_loader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      accuracy_metric=accuracy_metric,
      cuda_device=cuda_device,
      epoch_index=epoch_index,
      logging_frequency=logging_frequency
    )
   

    running_vloss = 0.0
    running_vacc = 0.0
    
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
      for i, vdata in enumerate(validation_loader):
        vinputs = vdata['image'].to(cuda_device)
        vlabels = vdata['label'].to(cuda_device)
        voutputs = model(vinputs)
        
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        
        vacc = accuracy_metric(voutputs, vlabels)
        running_vacc += vacc

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_vacc / (i + 1)
    print('LOSS train {} valid {} ACCURACY validation {}'.format(avg_loss, avg_vloss, avg_vacc))

    # Log the running loss averaged per batch
    # for both training and validation
    sum_writer.add_scalars(
      'Training vs. Validation Loss',
      { 'Training' : avg_loss, 'Validation' : avg_vloss },
      epoch_number + 1
    )
    sum_writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      model_path = 'model_{}_{}'.format(timestamp, epoch_number)
      torch.save(model.state_dict(), model_path)

    epoch_number += 1
    

def get_model_params(model):
  pp=0
  for p in list(model.parameters()):
    nn=1
    for s in list(p.size()):
      nn = nn*s
    pp += nn
  return pp

def evaluate_model(model, testing_fragment):
  model = model.to('cpu')
  model.eval()

  cum_error = 0

  inputs = testing_fragment['image']
  labels_true = testing_fragment['label']
  
  SET_SIZE = len(inputs)

  for i in range(SET_SIZE):
    input = inputs[i]
    label_true = labels_true[i]
    logits = model(input[None, ...]).detach().numpy()
    label_pred = np.argmax(logits)
    
    cum_error += abs(label_pred - label_true)
    print('index {}: true/predicted: {}/{}'.format(i, label_true, label_pred))
    
  error = cum_error / SET_SIZE

  accuracy = 1 - error

  print('testing accuracy: {}'.format(accuracy))