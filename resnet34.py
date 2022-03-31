import torch.nn as nn




class resi_block(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel, s, p):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.residual_function = nn.Sequential(
            
            nn.Conv2d(in_channels = in_channel, 
                      out_channels = out_channel, 
                      kernel_size = kernel, 
                      stride = s, 
                      padding = p, 
                      bias = False),
            
            nn.BatchNorm2d( num_features = out_channel ),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = out_channel, 
                      out_channels = out_channel, 
                      kernel_size = kernel, 
                      stride = s, 
                      padding = p, 
                      bias = False),
            
            nn.BatchNorm2d( num_features = out_channel ),
            )
        
    def forward(self, x):
        
        resi = self.residual_function(x)
        out = resi + x
        out = self.relu(out)
        return out

class resi_block_down(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel, s, p):
        super().__init__()
        
        self.relu = nn.ReLU()
        
        self.residual_function = nn.Sequential(
            
            nn.Conv2d(in_channels = int(in_channel/2), 
                      out_channels = out_channel, 
                      kernel_size = kernel, 
                      stride = (2,2), 
                      padding = p, 
                      bias = False),
            
            nn.BatchNorm2d( num_features = out_channel ),
            
            nn.ReLU(),
            
            nn.Conv2d(in_channels = out_channel, 
                      out_channels = out_channel, 
                      kernel_size = kernel, 
                      stride = s, 
                      padding = p, 
                      bias = False),
            
            nn.BatchNorm2d( num_features = out_channel ),
            )
        
        self.dimension = nn.Conv2d(in_channels = int(in_channel/2), 
                                   out_channels = out_channel, 
                                   kernel_size = (1,1), 
                                   stride = (2,2), 
                                   #padding = p, 
                                   bias = False)

    def forward(self, x):
        
        resi = self.residual_function(x)
        out = resi + self.dimension(x)
        out = self.relu(out)
        return out


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class res34(nn.Module):
    def __init__(self, num_module_list):
        super().__init__()
        
        self.base = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False),
                                  nn.BatchNorm2d(num_features = 64),
                                  nn.ReLU() ,
                                  nn.MaxPool2d(kernel_size = (3,3) , stride = (2,2), padding = 1, dilation = 1)
                                 )        
        
        self.resi1 = self.make_resi(num_modules = num_module_list[0], in_channel = 64, out_channel = 64, kernel = (3,3), s = (1,1), p = (1,1), down = False)
        self.resi2 = self.make_resi(num_modules = num_module_list[1], in_channel = 128, out_channel = 128, kernel = (3,3), s = (1,1), p = (1,1), down = True)
        self.resi3 = self.make_resi(num_modules = num_module_list[2], in_channel = 256, out_channel = 256, kernel = (3,3), s = (1,1), p = (1,1), down = True)
        self.resi4 = self.make_resi(num_modules = num_module_list[3], in_channel = 512, out_channel = 512, kernel = (3,3), s = (1,1), p = (1,1), down = True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = Flatten()
        self.fc = nn.Linear(512 , 100)
        
        
    def make_resi(self, num_modules, in_channel, out_channel, kernel, s, p, down):
        layers = list()
        if down == True:
            num_modules -= 1
            layers.append( resi_block_down ( in_channel, out_channel, kernel, s, p) )
            
        for i in range( num_modules ):
            layers.append( resi_block ( in_channel, out_channel, kernel, s, p) )

        return nn.Sequential( *layers )
        
        
    def forward(self, x):
        
        x = self.base(x)
        x = self.resi1(x)
        x = self.resi2(x)
        x = self.resi3(x)
        x = self.resi4(x)  
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str) 
    dir = parser.parse_args().dir

    import torch.onnx
    model = res34(num_module_list =[3,4,6,3])
    dummy_data = torch.empty(1, 3, 224, 224, dtype = torch.float32)
    torch.onnx.export(model, dummy_data, dir+"/model_architecture.onnx")
