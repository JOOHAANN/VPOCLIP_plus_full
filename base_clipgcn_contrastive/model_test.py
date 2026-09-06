import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
#from model import LeNet


def test_val_data_process():

    test_data = FashionMNIST(root="./data", train=False, download=True, transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]))
    
    test_data_loader = Data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

    return test_data_loader


def test_model_proccess(model, test_data_loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    test_corrects = 0.0
    test_num = 0
    
    with torch.no_grad():
        for test_data_x, test_data_y in test_data_loader:
            
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            
            model.eval()
            
            outputs = model(test_data_x)
            pre_label = torch.argmax(outputs, dim=1)
            
            test_corrects += torch.sum(pre_label == test_data_y)
            
            test_num += test_data_x.size(0)
            
    test_acc = test_corrects.double() / test_num
    print("Test Accuracy: {:.4f}".format(test_acc))
    
    

if __name__ == "__main__":
    model = LeNet()
    
    model.load_state_dict(torch.load("./model_weights/best_model_weights.pth"))
    
    test_data_loader = test_val_data_process()
    
    #test_model_proccess(model, test_data_loader)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    with torch.no_grad():
        for b_x, b_y in test_data_loader:
            
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            
            model.eval()
            
            outputs = model(b_x)
            pre_label = torch.argmax(outputs, dim=1)
            result = pre_label.item() == b_y.item()
            
            print("预测标签：", pre_label.item())
            print("真实标签：", b_y.item())
