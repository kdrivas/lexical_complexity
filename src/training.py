import numpy as np
import torch

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Function to calculate the accuracy of our predictions vs labels
def flat_metric(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return mean_absolute_error(pred_flat, labels_flat)

def forward_func_custom_bert(batch, device, model, additional_params):
    
    b_input_ids = batch['input_ids'].to(device)
    b_input_mask = batch['attention_mask'].to(device)
    b_labels = batch['labels'].to(device)
    b_positions =  batch['target_positions'].to(device)

    # Clear gradients
    model.zero_grad()        

    loss, logits = model(b_input_ids, 
                         b_positions,
                         token_type_ids=None, 
                         attention_mask=b_input_mask, 
                         labels=b_labels)
    
    return loss, logits

def train(device, model, loader, forward_func, optimizer, scheduler, additional_params={}):

    print('Training...')

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(loader):

        loss, logits = forward_func(batch, device, model, additional_params)

        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Compute gradients
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(loader)            

    print("")
    print("  Average training loss: {0:.6f}".format(avg_train_loss))
    
def evaluate(device, model, loader, forward_func, additional_params={}):
    print("")
    print("Running Validation...")
    
    model.eval()
    
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    all_logits = []
    training_stats = []
    # Evaluate data for one epoch
    ix = 0
    for batch in loader:
        ix += 1
        with torch.no_grad():        
            loss, logits = forward_func(batch, device, model, additional_params)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        
        b_labels = batch['labels'].to(device)
        label_ids = b_labels.to('cpu').numpy()
        
        val_metric = flat_metric(logits, label_ids)
        total_eval_accuracy += val_metric
        all_logits.append(logits.flatten()[0])

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(loader)
    print("  Metric: {0:.6f}".format(avg_val_accuracy))
    
    avg_val_loss = total_eval_loss / len(loader)
    
    print("  Validation Loss: {0:.6f}".format(avg_val_loss))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
        }
    )
    
    return training_stats, all_logits, total_eval_accuracy / len(loader)
