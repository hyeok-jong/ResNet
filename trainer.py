import time 
from tqdm import tqdm
import torch
import copy

# 현재의 lr을 출력합니다. scheduler를 사용 하기 때문에 바뀝니다. 따라서 매번 확인합니다.  
def get_lr(opt):         
    for param_group in opt.param_groups:
        return param_group['lr']
    



def trainer(model, params):
    num_epochs=params['num_epochs']
    loss_func=params["loss_func"]
    metric_function = params["metric_func"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    lr_scheduler=params["lr_scheduler"]
    result_dir=params["result_dir"]
    device = params["GPU"]
    


    loss_history = {'train': [], 'val': []}
    metric_history = {'train': [], 'val': []}
    

    best_model_weight = copy.deepcopy(model.state_dict())   
    # 아래에서 best parameter저장할때를 대비하여 미리 모양만 만들어 둔다. 공식문서에서 clone 대신 이거 사용함.                                             
    # https://discuss.pytorch.org/t/copy-deepcopy-vs-clone/55022
    
    best_loss = float('inf')
    # best model을 저장할때 기준이 loss value 이므로 미리 큰 값으로 설정 해 둔다.(작으면 좋은 거니까)

    start_time = time.time()

    for epoch in range(num_epochs):
        # 1-epoch이다.
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs-1, current_lr))

        
#-------------------------------------------------------------------------------------------------------
        model.train()   
        # nn.Module에 있다. 하는 역할이 크지는 않다. 하지만 필수 적인데 일단 이는 train과 validation을 구분하게 해준다.
        # parameter를 계산안하고 하고를 결정하는 것은 아니고, dropout같이 train과 validation에서 
        # 다른 연산을 하는 layer들에게 지금 뭘 하고 있는지 알려준다.
        
    
        running_loss = 0.0
        running_metric = 0.0
        # epoch 마다 0으로 만들어 준다.
    
        iteration_num_train = len(train_dl)
        # 이건 iteration 수이다.
        # 전체 데이터를 다 도는데 mini batch로 돌때 몇번 걸리는가이다.
        # len(train_dl.dataset)으로  전체 data의 개수이다.
        
        for inputs, targets in tqdm(train_dl):   
            # 1-batch train
            inputs , targets = inputs.to(device) , targets.to(device)
            outputs = model(inputs)
            
            loss_batch = loss_func(outputs, targets)
            metric_batch = metric_function(outputs, targets)
            
            opt.zero_grad()        # 이미 저장되어 있는 grad를 없애준다. 
            loss_batch.backward()  # autograd = True 되어있는 parameter들의 위치에서 grad를 계산한다.
            opt.step()             # update한다.
            
            running_loss += loss_batch.item()  # 이는 1-minibatch의 value이고 epoch이 될때까지 누적합을 계산한다.
            running_metric += metric_batch
                

            
        train_loss = running_loss / iteration_num_train     # 따라서 이 값이 1-epoch당 loss와 metric이다. 이때 굳이 loss에서 sum을 하였는데
        train_metric = running_metric / iteration_num_train # 이는 batch 별로 다 더하고 여기서 한번에 다음과 같이 나누는게 편해서이다.
                   
        loss_history['train'].append(train_loss)       # 매 epoch당 저장해 둔다.
        metric_history['train'].append(train_metric)
        
        
#-------------------------------------------------------------------------------------------------------
        model.eval()   # nn.Module에 있다
        
        with torch.no_grad():      
            # 이렇게 하여 with문 아래에서는 autograd = False가 되는데 이는 with문 아래에서만 일시적으로 그렇하다.
            # 아니면 transfer에서 사용하는 방법처럼 layer마다 grad를 off 해줘도 되는데 그럼 또 다시 켜줘야 하니까 이렇게 하는 것이 합리적이다.
            running_loss = 0.0
            running_metric = 0.0
        
            iteration_num_val = len(val_dl)
            
            for inputs, targets in val_dl:
                inputs , targets = inputs.to(device) , targets.to(device)
                outputs = model(inputs)
                
                loss_batch = loss_func(outputs, targets)
                metric_batch = metric_function(outputs, targets)
                
                running_loss += loss_batch.item()
                running_metric += metric_batch
                
            val_loss = running_loss / iteration_num_val
            val_metric = running_metric / iteration_num_val
            # 이거 loss 설정할때 mean으로 해줬으니 각 iteration마다는 평균으로 계산된다.
            # 하지만 이를 다 합쳤으니 iteration만큼 나눠줘야 한다.
                   
            loss_history['val'].append(val_loss)
            metric_history['val'].append(val_metric)

        # 중간에 끊길수 있으므로 best가 아니더라도 저장한다.
        # 공식 문서에도 모델을 직접 저장하면 수많은 이유로 오류가 발생할 수 있다고 한다.
        # 따라서 parameter만 저장한다.
        # model의 architecture는 저장하면 좋지만 사실 소스코드도 있다.
        # 또한 저장의 목적이 배포라면 모델도 저장하는 것이 맞지만
        # 단순히 break training 방지라면 굳이 할필요가 있을까 싶다.
        # 따라서 모델 저장하게 하긴 하지만 주석으로 남겨 놓는다!

        # 저장하는거 알고 싶다면 이 링크를 제일 추천한다
        # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch
        # 또는 kor. https://honeyjamtech.tistory.com/39
        # https://justkode.kr/deep-learning/pytorch-save



        if epoch % 5 == 0:
            
            torch.save(model.state_dict(), result_dir+'/trained_model_params_' + str(epoch) + '_epochs.pt')
            # torch.save(checkpoint_model, result_dir+'/trained_model_' + str(epoch) + '_epochs.pt')
            torch.save(opt.state_dict(), result_dir+'/trained_model_opt_' + str(epoch) + '_epochs.pt')

            print(f'Copied model parameter and opt with {epoch}th epoch')




        # Best model을 판단하고 저장하고 불러온다.
        if val_loss < best_loss:  
            best_loss = val_loss
            best_model_weight = copy.deepcopy(model.state_dict())    # 앞에서와 마찬가지로 parameter를 복사한다.
            # 참고로 deepcopy 하는 이유 https://androidkt.com/copy-pytorch-model-using-deepcopy-and-state_dict/
            torch.save(model.state_dict(), result_dir+'/best_model.pt')
            print('Copied best model weights!')
            print('Get best val_loss')

        lr_scheduler.step(val_loss)

        print('train loss: %.6f, train accuracy: %.2f' %(train_loss, 100*train_metric))
        print('val loss: %.6f, val accuracy: %.2f' %(val_loss, 100*val_metric))
        print('time: %.4f min' %((time.time()-start_time)/60))
        print('-'*50)

        # 매 epoch마다 loss와 metric저장 -> 최종으로 반환하기는 하는데 중간에 끊기는거 대비

        with open(result_dir+ '/train_loss.txt','w',encoding='UTF-8') as f:
            for loss in loss_history['train']:
                f.write(str(epoch)+" : "+str(loss)+'\n')
        with open(result_dir+ '/val_loss.txt','w',encoding='UTF-8') as f:
            for loss in loss_history['val']:
                f.write(str(epoch)+" : "+str(loss)+'\n')
        with open(result_dir+ '/train_metric.txt','w',encoding='UTF-8') as f:
            for metric in metric_history['train']:
                f.write(str(epoch)+" : "+str(metric)+'\n')
        with open(result_dir+ '/val_metric.txt','w',encoding='UTF-8') as f:
            for metric in metric_history['val']:
                f.write(str(epoch)+" : "+str(metric)+'\n')


    model.load_state_dict(best_model_weight)  


    ######### 항상 마지막 epoch이 best가 아니므로 test를 위해 best의 parameter를 불러준다.
    # 이때 불러줄때 이미 짜여진 model의 class에 저렇게 불러줘야 한다.
    # 그러면 parameter 자리에 알아서 잘 들어간다.

    return model, loss_history, metric_history
