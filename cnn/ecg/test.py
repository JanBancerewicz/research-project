import torch


def test_model(device, model, test_loader, criterion):
    model.eval()  # Ustawienie modelu w tryb ewaluacji
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 256)  # Dopasowanie do wyjścia modelu

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            prediction = torch.sigmoid(outputs) > 0.5
            correct_predictions += (prediction == labels).sum().item()
            total_predictions += labels.numel()

    avg_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
