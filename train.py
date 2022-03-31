from os import lseek
from dataloader import make_dataloader
from resnet34 import res34, initialize_weights
from trainer import trainer
import argparse
import loss 
from metric import metric_function
import matplotlib.pyplot as plt


import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau 

# optimizer와 scheduler는 model이 선언된 이후 parameter를 입력해 줘야 하므로 여기서 정의 합니다.  


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help='Directory where ur datasets exists.   e.g) /home/mskang/hyeokjong/ResNet/cifar-100-python')   # dataset의 위치를 받습니다.
    parser.add_argument("--result_dir", type = str, help ="Directory where results will be saved")
    parser.add_argument("--GPU", type = int, help = "GPU ID e.g) 1")
    parser.add_argument("--batch_size", type = int, help = "mini-batch size")
    parser.add_argument("--epoch", type = int, help = "training epochs")
    return parser.parse_args()





if __name__ == "__main__":   # 이 파일 자체를 어디에서 import하지는 않으므로 굳이 if문 달 필요는 없음. 다면 연습.
    args = args()
    train_dl, val_dl, test_dl = make_dataloader(args.data_dir, args.batch_size)
    device = "cuda:" + str(args.GPU)
    model = res34(num_module_list =[3,4,6,3]).to(device)
    model.apply(initialize_weights)

    # optimizer 와 scheduler를 설정한다.
    #opt = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    #opt = torch.optim.Rprop(model.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    #opt = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False) 
    
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)



    params_train = {
    'num_epochs':args.epoch,
    'optimizer':opt,
    'loss_func':loss.loss_function,  # imported from loss.py
    'train_dl':train_dl,
    'val_dl':val_dl,
    'lr_scheduler':lr_scheduler,
    'result_dir': args.result_dir,
    'GPU' : "cuda:"+ str(args.GPU),
    "metric_func" : metric_function }

    model, loss_history, metric_history = trainer(model, params_train)


    # 최종 학습이 끝난 후 저장합니다.
    result_dir = args.result_dir
    torch.save(model.state_dict(), result_dir+'/trained_model_params_final_' + str(args.epoch) + '_epochs.pt')
    # torch.save(checkpoint_model, result_dir+'/trained_model_final_' + str(args.epoch) + '_epochs.pt')
    torch.save(opt.state_dict(), result_dir+'/trained_model_opt_fianl_' + str(args.epoch) + '_epochs.pt')
    print(f'Copied model parameter and opt with {args.epoch}th epoch')

    num_epochs=params_train["num_epochs"]

    # plot loss progress
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_history["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_history["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.savefig(result_dir + "/loss_hist.png")
    

    plt.title("Train-Val Metric")
    plt.plot(range(1,num_epochs+1),metric_history["train"],label="train")
    plt.plot(range(1,num_epochs+1),metric_history["val"],label="val")
    plt.ylabel("Metric")
    plt.xlabel("Training Epochs")
    plt.savefig(result_dir + "/metric_hist.png")

    







    # test dataset

    running_loss = 0.0
    running_metric = 0.0
    loss_func = params_train['loss_func']
    with torch.no_grad():      
        running_loss = 0.0
        running_metric = 0.0
            
        len_data_test = len(test_dl.dataset)
                
        for inputs, targets in test_dl:
            inputs , targets = inputs.to(device) , targets.to(device)
            outputs = model(inputs)
                    
            loss_batch = loss_func(outputs, targets)
            metric_batch = metric_function(outputs, targets)
                    
            running_loss += loss_batch.item()
            running_metric += metric_batch
        test_loss = running_loss / len_data_test
        test_metric = running_metric / len_data_test
    print('test loss: %.6f, test accuracy: %.2f' %(test_loss, 100*test_metric))























