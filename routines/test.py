import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F


def test_classify(model, test_loader, device, criterion):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        # outputs = model(feats)[1]
        outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


def test_model(model, test_loader, device):

    # final_output = []

    with torch.no_grad():
        model.eval()
        model.to(device)

        for batch_num, feats in enumerate(test_loader):
            feats = feats.to(device)
            outputs = model(feats)
            m = torch.nn.Softmax(dim=1)
            prob = m(outputs)

            # final_output.append(prob.cpu().numpy())

        return prob.cpu().numpy()
    # return final_output
