def train_loop(dataloader, loss_fn, optimizer, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch+1) % (num_batches//10) == 0:
            loss, current = loss.item(), (batch+1)*batch_size
            print(f"Loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, loss_fn, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0

    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()

    avg_loss = test_loss / num_batches
    correct /= size

    print(f"Average test loss: {avg_loss:>7f}, Accuracy: {correct*100:>0.1f}%")