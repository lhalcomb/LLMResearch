"""
This file contains some utility functions for easier compatibility and avoiding 
long files throughout the project and next project using this code.

Process is in 3 steps is as follows: (also illustrated in figure 5.3 on page 131)

1. the tokenizer converts input text into a series of token IDs 

2. the model receives these token IDs and generates corresponding logits, which are 
vectors representing the probability distribution for each token in the vocabulary

3. these logits are converted back into token IDs, which the tokenizer
decodes into human-readable text, completing the cycle from textual input to textual output.


"""

import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
        """
        idx is a (batch, n_tokens)
        array of indices in the
        current context.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]

            with torch.no_grad():
                logits = model(idx_cond)
        """
        Line: 417 logits = logits[:, -1, :]
            Focuses only on the last time step,
        so that (batch, n_token, vocab_size)
        becomes (batch, vocab_size)
        """
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1) #probas has shape (batch, vocab_size).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  #idx_next has shape (batch, 1).
        idx = torch.cat((idx, idx_next), dim=1) #Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)

        return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """ Calculates the loss for a single batch """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
    logits.flatten(0, 1), target_batch.flatten()
    )

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Function to compute the training and validation loss"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader) #iterates over all batches if no fixed num_batches is specified
    else:
        num_batches = min(num_batches, len(data_loader))
        #^^ reduces the number of batches to match the total number of batches in the data loader 
        #^^ if num_batches exceeds the number of batches in the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
                )
            total_loss += loss.item() #accumulate the loss for each batch
        else:
            break
    return total_loss / num_batches #average the loss over all the batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    """
    Evaluate the model on training and validation data loaders and return their average losses

    :param model: The GPTModel to evaluate
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param device: Device to run evaluation on (e.g., 'cpu' or 'cuda')
    :param eval_iter: Number of batches to use for evaluation
    :return: Tuple containing average training loss and validation loss
    :rtype: tuple[float, float]

    """

    model.eval() #dropout disabled here for stable reproducible results
    with torch.no_grad(): #disables gradient tracking, not useful in evaluation
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
            )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()

    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Generate a text sample from the model using the given context and print the result

    :param model: The GPTModel for text generation
    :param tokenizer: Tokenizer for encoding and decoding text
    :param device: Device to run the model on (e.g., 'cpu' or 'cuda')
    :param start_context: Initial text context to begin generation
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " ")) 
    model.train()

def train_model_simple(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], [] #lists to track losses and tokens seen
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs): #starts train loop
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() #resets loss gradients from prev batch
            loss = calc_loss_batch(  
            input_batch, target_batch, model, device
            )
            loss.backward() #calculates loss gradients
            optimizer.step() #updates model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0: #optional eval step
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                     )
        generate_and_print_sample( #prints sample after each epoch
            model, tokenizer, device, start_context
            )
        
    return train_losses, val_losses, track_tokens_seen