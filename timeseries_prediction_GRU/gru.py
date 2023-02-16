# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRU_NN(nn.Module):
    
    def __init__(self, dropout, input_dim, output_dim, hidden_dim, num_layers, device=device):
        super(GRU_NN,self).__init__()
        
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        if num_layers > 1:
            self.dropout = dropout
        else:
            self.dropout = 0
        self.num_layers = num_layers
        
        #initialize weights from normal distribution
        self.apply(self._init_weights)

        # define LSTM layer
        self.gru = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size,
                            num_layers = self.num_layers, batch_first=True, 
                            dropout = self.dropout)
        
        #self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)
          
        
    def forward(self,x, h):
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, hn = self.gru(x, h)

        # Index hidden state of last time step 
        #out = self.fc(self.relu(out[:, -1])) 
        out = self.fc(hn[-1,:]) 
        return out, hn
    
    def init_hidden(self, batch_size):
        
        weight = next(self.parameters()).data        
        h = weight.new(
            self.num_layers, batch_size, self.hidden_size).zero_().to(device)
                
        return h
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(mean=0.0, std=1.0)
            #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                m.bias.data.zero_()
                #nn.init.zeros_(m.bias)

    
    def train_model(self,criterion,optimizer,scheduler,  num_epochs, model, training_loader, x_valid, y_valid,  model_name, batch_size):
        print ("training")  
        
        
        epoch_loss = []
        for epoch in range(num_epochs):  
            h = model.init_hidden(batch_size)
            avg_loss = 0
            counter = 0
            
            
            for i, batch in enumerate(training_loader):
                counter += 1

                h = h.data
                
                features, labels = batch
                features.to(device) 
                
                optimizer.zero_grad()
                outputs, h = model(features, h)
                
                # obtain the loss function
                loss = criterion(outputs, labels)
                loss.backward()
                
                #The norm is computed over all gradients together, as if they were concatenated into a single vector. All of the gradient coefficients are multiplied by the same clip_coef.
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                # It clipping the derivatives of the loss function to have a given value if a gradient value is less than a negative threshold or more than the positive threshold. Threshhold given by clip_value
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.)
                #register a backward hook
                #for p in model.parameters():
                #   p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))   
                 
                optimizer.step()
                scheduler.step()
                # add the mini-batch training loss to epoch loss
                avg_loss += loss.item()
                epoch_loss.append(loss.cpu().item())
                # ===================log======================== 
             
                # compute the epoch training loss
            #if epoch%10==0:
            #print('epoch [{}/{}], loss:{:.6f}'
            #      .format(epoch + 1, num_epochs, loss.item(),  avg_loss/counter))#[0]))
            
            # Get validation losses
            #valid_results, targets, error = self.predict(model, x_valid, y_valid)
            #print('valid epoch [{}/{}], loss:{:.6f}'
            #  .format(epoch + 1, num_epochs, error))#[0]))
            
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss/(i+1)
                }, model_name)
            
            with torch.no_grad():
                h = model.init_hidden(x_valid.shape[0])
                valid_out, h = model(x_valid,h)
                vall_loss = criterion(valid_out, y_valid)
                #scheduler.step(vall_loss)

            if epoch % 10 == 0:
                print("Epoch: %d, loss: %1.5f avg_loss: %1.5f valid loss:  %1.5f " %(epoch, loss.cpu().item(), avg_loss/(i+1), vall_loss.cpu().item()))
             
        #print("epoch_loss: ",list(epoch_loss))
            

        
    
    
    def predict(self, model, x_test, y_test):
        results = []; targets = []
        
        #set to prediction mode: ignores dropout layers
        with torch.no_grad():
            h = model.init_hidden(x_test.shape[0])
            output, h = model(x_test, h)
            # Get predictions from the maximum value
            results.append((output.cpu().detach().numpy()).reshape(-1))
            targets.append((y_test.cpu().detach().numpy()).reshape(-1))
            
        # Symmetric Mean Absolute Percentage Error --> percentage measuring the amount of error
            
        sMAPE = 0
        for i in range(len(results)):
            sMAPE += np.mean(abs(results[i]-targets[i])/(targets[i]+results[i])/2)/len(results)
        print("sMAPE: {}%".format(sMAPE*100))
        
        error = np.mean(abs(np.array(results)-np.array(targets)))
        return results, targets, sMAPE    

        
    def run_predict(self,x_test, y_test, model_name, model):   
            
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        #loss = checkpoint['loss']

        result, target, err = model.predict(model, x_test, y_test)
        
        return result, target
        
        

    
    
    

