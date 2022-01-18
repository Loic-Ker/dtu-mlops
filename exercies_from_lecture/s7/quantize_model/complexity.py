import torchvision.models as models
from ptflops import get_model_complexity_info

net = models.mobilenet_v3_small()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True)

f= open("mobilenetv3.txt","w+")
f.write('{:<30}  {:<8}'.format('Computational complexity: ', macs))
f.write('{:<30}  {:<8}'.format('Number of parameters: ', params))
f.close()

net = models.resnet152()
macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True)

f= open("resnet152.txt","w+")
f.write('{:<30}  {:<8}'.format('Computational complexity: ', macs))
f.write('{:<30}  {:<8}'.format('Number of parameters: ', params))
f.close()