import torch  

def metric_function(output, target):
    _, argmax = torch.max(output, dim = 1)
    corrects = (argmax == target).sum()
    return corrects.item()/target.shape[0]    # torch로 나와서 item만 뽑습니다.
