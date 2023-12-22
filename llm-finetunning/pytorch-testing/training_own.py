def train_one_epoch(epoch_index, tb_writer, logging_frequency):
  running_loss = 0.
  running_accuracy = 0.
  last_loss = 0.

  # Here, we use enumerate(training_loader) instead of
  # iter(training_loader) so that we can track the batch
  # index and do some intra-epoch reporting
  for i, data in enumerate(training_loader):
  
    # Every data instance is an input + label pair
    inputs = data['image'].to(device)
    labels = data['label'].to(device)
  
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
      tb_writer.add_scalar('Loss/train', last_loss, tb_x)
      running_loss = 0.
      running_accuracy = 0.
  
  return last_loss


def train_many_epochs(epochs, writer, logging_frequency):
  best_vloss = 1_000_000.

  for epoch_number in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer, logging_frequency)

    running_vloss = 0.0
    running_vacc = 0.0
    
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
      for i, vdata in enumerate(validation_loader):
        vinputs = vdata['image'].to(device)
        vlabels = vdata['label'].to(device)
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
    writer.add_scalars(
      'Training vs. Validation Loss',
      { 'Training' : avg_loss, 'Validation' : avg_vloss },
      epoch_number + 1
    )
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      model_path = 'model_{}_{}'.format(timestamp, epoch_number)
      torch.save(model.state_dict(), model_path)

    epoch_number += 1