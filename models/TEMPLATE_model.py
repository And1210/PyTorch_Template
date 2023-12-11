import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel
from sklearn.metrics import accuracy_score
from collections import OrderedDict

#PyTorch defined model
class TEMPLATE(nn.Module):
    """basenet for fer2013"""
    def __init__(self, in_channels=1, num_classes=7):
        super(TEMPLATE, self).__init__()
        norm_layer = nn.BatchNorm2d

        #Here is where you define your architecture (by defining layers)
        #Everything in this function after this is all custom
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residual_1 = ResidualUnit(in_channels=64, out_channels=256)
        self.residual_2 = ResidualUnit(in_channels=256, out_channels=512)
        self.residual_3 = ResidualUnit(in_channels=512, out_channels=1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 7)

    #The function which feeds forwards data into different layers
    #Use the above defined layers here on input data
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#The abstract model class, uses above defined class and is used in the train script
class TEMPLATEmodel(BaseModel):
    """basenet for fer2013"""

    def __init__(self, configuration, in_channels=1, num_classes=7):
        super().__init__(configuration)

        #Initialize model defined above
        self.model = TEMPLATE(in_channels, num_classes)
        self.model.cuda()

        #Define loss function
        self.criterion_loss = nn.CrossEntropyLoss().cuda()
        #Define optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=configuration['lr'],
            momentum=configuration['momentum'],
            weight_decay=configuration['weight_decay']
        )

        #Need to include these arrays with the optimizers and names of loss functions and models
        #Will be used by other functions for saving/loading
        self.optimizers = [self.optimizer]
        self.loss_names = ['total']
        self.network_names = ['model']

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []

    #Calls the models forwards function 
    def forward(self):
        x = self.input
        self.output = self.model.forward(x)
        return self.output

    #Computes the loss with the specified name (in this case 'total')
    def compute_loss(self):
        self.loss_total = self.criterion_loss(self.output, self.label)

    #Compute backpropogation for the model
    def optimize_parameters(self):
        self.loss_total.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    #Test function for the model
    def test(self):
        super().test() # run the forward pass

        # save predictions and labels as flat tensors
        self.val_images.append(self.input)
        self.val_predictions.append(self.output)
        self.val_labels.append(self.label)

    #Should be run after each epoch, outputs accuracy
    def post_epoch_callback(self, epoch, visualizer):
        self.val_predictions = torch.cat(self.val_predictions, dim=0)
        predictions = torch.argmax(self.val_predictions, dim=1)
        predictions = torch.flatten(predictions).cpu()

        self.val_labels = torch.cat(self.val_labels, dim=0)
        labels = torch.flatten(self.val_labels).cpu()

        self.val_images = torch.squeeze(torch.cat(self.val_images, dim=0)).cpu()

        # Calculate and show accuracy
        val_accuracy = accuracy_score(labels, predictions)

        metrics = OrderedDict()
        metrics['Accuracy'] = val_accuracy

        if (visualizer != None):
            visualizer.plot_current_validation_metrics(epoch, metrics)
        print('Validation accuracy: {0:.3f}'.format(val_accuracy))

        # Here you may do something else with the validation data such as
        # displaying the validation images or calculating the ROC curve

        self.val_images = []
        self.val_predictions = []
        self.val_labels = []


if __name__ == "__main__":
    net = TEMPLATEmodel().cuda()
    from torchsummary import summary

    print(summary(net, input_size=(1, 48, 48)))
