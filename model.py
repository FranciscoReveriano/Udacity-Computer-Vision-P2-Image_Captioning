import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # Hidden States
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Embedding Layer
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        # LSTM Model
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size, 
                            num_layers=num_layers,
                            bias=True,
                            batch_first=True
                            )
        # Final Layer
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        embeddings = self.word_embeddings(captions[:,:-1])
        
        features = features.view(len(features),1,-1)
        
        inputs = torch.cat((features, embeddings.float()), dim=1)
        
        outputs, lstm_hidden = self.lstm(inputs)
        
        outputs = self.linear(outputs)
        
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        output = []
        word_count = 0
        hidden = None
        while True:
            
            if hidden == None:
                hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
                out_lstm, hidden = self.lstm(inputs, hidden)
            else:
                out_lstm, hidden = self.lstm(inputs, hidden)
            
            out = self.linear(out_lstm)
            out = out.squeeze(1)
            _, max_indice = torch.max(out, dim=1)
            
            output.append(max_indice.cpu().numpy()[0].item())
                       
            if max_indice == 1:
                break
            inputs = self.word_embeddings(max_indice)
            inputs = inputs.unsqueeze(1)
            
        return output
        
        