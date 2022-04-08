from torchvision.transforms.transforms import ToTensor

# from sklearn.metrics import classification_report
import torch
import wandb
from tqdm import tqdm


target = []
predictions = []


# Функция обучения сети
# def train_model(model, loss, optimizer, scheduler, num_epochs, path_weigh_save):
def train_model(model, train_dataloader, test_dataloader, val_dataloader, loss, optimizer, scheduler, num_epochs,
                    path_weigh_save, model_name:str):

    '''
    первый выход - после 2 слоя в резнете
    второй выход - последний выход

    '''
    # wandb.init(
    #     project="googlecolab_ResNet2_outputs_common-loss_output-after-2layer_avg_15tpoch_bs=23",
    #     entity="maria_mikhalkova",
    #     config={
    #         "epochs": 15,
    #         "batch_size": 23,
    #         "lr": 1e-4,
    #         # "dropout": random.uniform(0.01, 0.80),
    #     })
    # config = wandb.config

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    running_acc_train_2out = 0.
    running_acc_train_1out = 0.
    train_acc_2out = []
    train_acc_1out = []

    val_loss_2out = []
    val_loss_1out = []
    val_acc_1out = []
    val_acc_2out = []
    running_acc_val_2out = 0.
    running_acc_val_1out = 0.
    running_acc2 = 0.

    test_loss_2out = []
    test_loss_1out = []
    test_acc_1out = []
    test_acc_2out = []
    running_acc_test_2out = 0.
    running_acc_test_1out = 0.
    # running_acc2 = 0.

    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            elif phase == 'val':
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode
            else:
                dataloader = test_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss_2out = 0.
            running_loss_1out = 0.
            running_acc_2out = 0.
            running_acc_1out = 0.

            # Iterate over data.
            for inputs, labels, paths in tqdm(dataloader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autograd.set_detect_anomaly(True):
                        # u = Tensor()
                        # preds, u1 = model(inputs.cuda(),u)
                        preds_2, preds_1 = model(inputs)
                        print(preds_1.shape)
                        if model_name == 'Entropia_2output':
                            all_loss_value_1_out = []
                            for i in (range(preds_1.shape[1])):
                                pred_1 = preds_1[:, i, :]
                                all_loss_value_1_out.append(loss(pred_1, labels))
                            loss_value_1_out = min(all_loss_value_1_out)
                            all_loss_value_1_out = torch.stack(all_loss_value_1_out)
                            arg_loss_value_1_out = all_loss_value_1_out.argmin(dim=0)
                            preds_class_1_out = preds_1[:, arg_loss_value_1_out, :].argmax(dim=1)
                        elif model_name == 'Entropia_2output_2':
                            loss_value_1_out = loss(preds_1, labels)
                            preds_class_1_out = preds_1.argmax(dim=1)
                        else:

                            loss_value_1_out = loss(pred_1, labels)
                            preds_class_1_out = preds_1.argmax(dim=1)

                        loss_value_2_out = loss(preds_2, labels)
                        loss_common = 0 * loss_value_1_out + loss_value_2_out
                        preds_class_2_out = preds_2.argmax(dim=1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # loss_value_2_out.backward()
                            # autograd.set_detect_anomaly(True)
                            loss_common.backward()
                            optimizer.step()
                            # train_loss.append(loss_value_2_out.item())
                            train_loss.append(loss_common.item())
                            # classific_report=classification_report(labels.data.cuda(), preds_class, target_names=labels.data.cuda())
                            running_acc_train_2out += (preds_class_2_out == labels.data).float().mean()
                            running_acc_train_1out += (preds_class_1_out == labels.data).float().mean()
                            train_acc_2out.append(running_acc_train_2out)
                            train_acc_1out.append(running_acc_train_1out)

                            metrics = {"train/train_loss": train_loss,
                                       "train/running_acc_train_2out": running_acc_train_2out,
                                       "train/running_acc_train_1out": running_acc_train_1out,
                                       "train/train_acc_2out": train_acc_2out,
                                       "train/train_acc_1out": train_acc_1out,
                                       "train/epoch": epoch,
                                       }

                            # wandb.log(metrics)

                        elif phase == 'val':

                            val_loss_2out.append(loss_value_2_out.item())
                            val_loss_1out.append(loss_value_1_out.item())
                            running_acc_val_1out += (preds_class_1_out == labels.data).float().mean()
                            running_acc_val_2out += (preds_class_2_out == labels.data).float().mean()
                            val_acc_2out.append(running_acc_val_2out)
                            val_acc_1out.append(running_acc_val_1out)

                            val_metrics = {
                                "val/val_loss_2out": val_loss_2out,
                                "val/val_loss_1out": val_loss_1out,
                                "val/running_acc_val_1out": running_acc_val_1out,
                                "val/running_acc_val_2out": running_acc_val_2out,
                                "val/val_acc_2out": val_acc_2out,
                                "val/val_acc_1out": val_acc_1out,
                                "val/epoch": epoch,
                            }

                            # wandb.log(val_metrics)

                        else:
                            test_loss_2out.append(loss_value_2_out.item())
                            test_loss_1out.append(loss_value_1_out.item())
                            running_acc_test_1out += (preds_class_1_out == labels.data).float().mean()
                            running_acc_test_2out += (preds_class_2_out == labels.data).float().mean()
                            test_acc_2out.append(running_acc_test_2out)
                            test_acc_1out.append(running_acc_test_1out)

                            test_metrics = {
                                "test/test_loss_2out": test_loss_2out,
                                "test/test_loss_1out": test_loss_1out,
                                "test/running_acc_test_1out": running_acc_test_1out,
                                "test/running_acc_test_2out": running_acc_test_2out,
                                "test/test_acc_2out": test_acc_2out,
                                "test/test_acc_1out": test_acc_1out,
                                "test/epoch": epoch,
                            }

                            # wandb.log(test_metrics)

                    # statistics
                    running_loss_2out += loss_value_2_out.item()
                    running_loss_1out += loss_value_1_out.item()

                    running_acc_2out += (preds_class_2_out == labels.data).float().mean()
                    running_acc_1out += (preds_class_1_out == labels.data).float().mean()

                epoch_loss_2out = running_loss_2out / len(dataloader)
                epoch_acc_2out = running_acc_2out / len(dataloader)

                epoch_loss_1out = running_loss_1out / len(dataloader)
                epoch_acc_1out = running_acc_1out / len(dataloader)

                # print('{} 2 OUTPUT: Loss: {:.4f} Acc: {:.4f}  1 OUTPUT: Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss_2out, epoch_loss_2out,  epoch_loss_1out, epoch_loss_1out), flush=True)
                print('{} 2 OUTPUT: Loss: {:.4f} Acc: {:.4f}  1 OUTPUT: Loss: {:.4f} Acc: {:.4f}'.format(phase,
                                                                                                         epoch_loss_2out,
                                                                                                         epoch_acc_2out,
                                                                                                         epoch_loss_1out,
                                                                                                         epoch_acc_1out),
                      flush=True)

                train_acc_cpu = torch.tensor(train_acc_2out).cpu()
                # writer.add_scalar('train_accuracy', scalar_value=epoch_acc, global_step=epoch)

                if (phase == 'val'):
                    val_acc_cpu = torch.tensor(val_acc_2out).cpu()
                    # writer.add_scalar('val_accuracy', scalar_value=epoch_acc_2out, global_step=epoch)

                torch.save(model.state_dict(), path_weigh_save + str(epoch))
    # wandb.watch(model)
    # wandb.finish()
    return model, train_loss, val_loss_2out, train_acc_2out, val_acc_2out, train_acc_1out, val_acc_1out
