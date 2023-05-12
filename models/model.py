import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.functions import ReverseLayerF

class Informer(nn.Module):
    def __init__(self, args, OP_features, c_out, seq_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        
        if OP_features == True:
            enc_in, dec_in = 20,20

        elif OP_features == False:
            enc_in, dec_in = 14,14
            
        
        self.args = args
        self.seq_len = seq_len
        self.attn = attn
        self.output_attention = output_attention
        
        self.dropout = nn.Dropout(self.args.dropout)  
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        
        # Private Encoder
        self.encoder_private = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Share Encoder
        self.encoder_share = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        '''
        #FCL RUL predictor
        self.RUL_predictor = nn.Sequential()
        self.RUL_predictor.add_module('r_fc1',nn.Linear(d_model,256))
        self.RUL_predictor.add_module('r_bn1', nn.BatchNorm1d(self.seq_len))
        self.RUL_predictor.add_module('r_relu1', nn.ReLU(True))
        self.RUL_predictor.add_module('r_drop1', nn.Dropout())
        self.RUL_predictor.add_module('r_fc2', nn.Linear(256, 128))
        self.RUL_predictor.add_module('r_bn2', nn.BatchNorm1d(self.seq_len))
        self.RUL_predictor.add_module('r_relu2', nn.ReLU(True))
        self.RUL_predictor.add_module('r_drop2', nn.Dropout())
        self.RUL_predictor.add_module('r_fc3', nn.Linear(128, 64))
        self.RUL_predictor.add_module('r_fc4', nn.Linear(64, 1))
        '''
        
        #decoder RUL predictor
        # Decoder   
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),  #d_model
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),  #d_model
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection1 = nn.Linear(d_model, 128, bias=True) #d_model
        self.projection2 = nn.Linear(128, 64, bias=True) #d_model
        self.projection3 = nn.Linear(64, 1, bias=True)
        self.relu = nn.ReLU(True)
        
        
        #domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(d_model, 128))
        self.domain_classifier.add_module('c_bn1', nn.BatchNorm1d(18))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_drop1', nn.Dropout())
        self.domain_classifier.add_module('c_fc2', nn.Linear(128, 64))
        self.domain_classifier.add_module('c_bn2', nn.BatchNorm1d(18))
        self.domain_classifier.add_module('c_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('c_fc3', nn.Linear(64, 1))
        self.domain_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        
        #domain discriminator
        self.domain_discriminator = nn.Sequential()
        self.domain_discriminator.add_module('d_fc1', nn.Linear(d_model, 128))
        self.domain_discriminator.add_module('d_bn1', nn.BatchNorm1d(18))
        self.domain_discriminator.add_module('d_relu1', nn.ReLU(True))
        self.domain_discriminator.add_module('d_fc2', nn.Linear(128, 64))
        self.domain_discriminator.add_module('d_bn2', nn.BatchNorm1d(18))
        self.domain_discriminator.add_module('d_relu2', nn.ReLU(True))
        self.domain_discriminator.add_module('d_fc3', nn.Linear(64, 1))
        self.domain_discriminator.add_module('d_softmax', nn.LogSoftmax(dim=1))
 
        
        
    def forward(self, x, flag, alpha,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        #embedding
        x_emd = self.enc_embedding(x) 
  
        #private encoder & share encoder
        private_feature, private_attns = self.encoder_private(x_emd, attn_mask=enc_self_mask)  #out size  bs*seq_len/2*d_model  64*24*512
        share_feature, share_attns = self.encoder_share(x_emd, attn_mask=enc_self_mask) 
        
        #RUL predictor
        #features = torch.cat((private_feature,share_feature),dim=1)  
        #Y_RUL = self.RUL_predictor(features)
        Y_RUL = self.decoder(private_feature,share_feature, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        #Y_RUL = self.decoder(x_emd,features, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        Y_RUL = self.dropout(self.projection1(Y_RUL))
        Y_RUL = self.dropout(self.projection2(Y_RUL))
        Y_RUL = self.projection3(Y_RUL)
        Y_RUL = Y_RUL[:,-1,:]
        
        #domain classifier
        class_output = self.domain_classifier(private_feature)
        
        #domain discriminator
        if flag in ['train_source','train_target']:
            share_feature = ReverseLayerF.apply(share_feature, alpha)
        domain_output = self.domain_discriminator(share_feature)
                  
        return Y_RUL, class_output, domain_output, share_feature, private_feature


        
        

