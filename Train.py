from torchvision.transforms.transforms import ToTensor

# from sklearn.metrics import classification_report
import torch
from torch import Tensor
from tqdm import tqdm

target = []
predictions = []


# Функция обучения сети
def train_model(model, train_dataloader, train_dataset, val_dataloader, loss, optimizer, scheduler, num_epochs, path_weigh_save):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    running_acc1 = 0.
    running_acc2 = 0.
    for epoch in range(num_epochs):
        # TODO: изменить макспулинг
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels, paths in tqdm(dataloader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                optimizer.zero_grad()


                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    u = Tensor()
                    # preds, u1 = model(inputs.cuda(),u)
                    # preds = model(inputs.cuda())
                    preds, u = model(inputs)
                    # print(torch.tensor(preds).shape)
                    print(preds[0])
                    # print("11111111111")
                    # print(u)
                    # loss_value = loss(preds[1], labels.cuda())
                    # preds_class = preds[1].argmax(dim=1)
                    # loss_value = loss(preds, labels.cuda())
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                        train_loss.append(loss_value.item())
                        # classific_report=classification_report(labels.data.cuda(), preds_class, target_names=labels.data.cuda())
                        # running_acc1 += (preds_class == labels.data.cuda()).float().mean()
                        running_acc1 += (preds_class == labels.data).float().mean()
                        train_acc.append(running_acc1)

                    else:

                        val_loss.append(loss_value.item())
                        # running_acc2 += (preds_class == labels.data.cuda()).float().mean()
                        running_acc2 += (preds_class == labels.data).float().mean()
                        val_acc.append(running_acc2)

                # statistics
                running_loss += loss_value.item()

                # running_acc += (preds_class == labels.data.cuda()).float().mean()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

            train_acc_cpu = torch.tensor(train_acc).cpu()
            # writer.add_scalar('train_accuracy', scalar_value=epoch_acc, global_step=epoch)

            if (phase == 'val'):
                val_acc_cpu = torch.tensor(val_acc).cpu()
                # writer.add_scalar('val_accuracy', scalar_value=epoch_acc, global_step=epoch)

            torch.save(model.state_dict(), path_weigh_save + str(epoch))

    return model, train_loss, val_loss, train_acc, val_acc
