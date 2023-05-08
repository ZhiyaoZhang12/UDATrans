import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, args, OP_features, c_out, seq_len, label_len, out_len, is_perception,
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
        self.pred_len = out_len
        self.label_len = label_len
        self.is_perception = is_perception   
        self.attn = attn
        self.output_attention = output_attention
        
        self.dropout = nn.Dropout(self.args.dropout)  

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
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
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        d_fmap = 256   #bsx(*)x512  -->  bsx(*)x256
        self.features = nn.Linear(d_model, d_fmap, bias=True)  #fearure map
        
        if self.args.fmap == True:
            d_projection,d_rul_in,d_soh_in = d_fmap,d_fmap,d_fmap  ##bsx(*)x256
        elif self.args.fmap == False:
            d_projection,d_rul_in,d_soh_in = d_model,d_model,d_model  ##bsx(*)x512  
        if self.args.MTLtype == 'hierarchy':
            d_projection = d_model   #层次的时候，一定有decoder， 此时 d_projection只等于d_model
        

        #HI
        self.projection = nn.Linear(d_projection, c_out, bias=True)
        
        #RUL
        hidden_size_rul1, hidden_size_rul2 = 128, 64    #d_model 512
        self.FC1 = nn.Linear(d_rul_in, hidden_size_rul1, bias=True)  
        self.FC2 = nn.Linear(hidden_size_rul1, hidden_size_rul2, bias=True)
        self.FC3 = nn.Linear(hidden_size_rul2, 1, bias=True)
        
        ##MTL  seq_len48对应24    32对应16   36对应18     bs*seq_len*d_model
        #enc_out_size =  int(self.seq_len/2)   
        
        #SOH
        hidden_size_soh = 128
        self.SOH_fcl = nn.Linear(d_soh_in,hidden_size_soh)  
        self.SOH_fc2 = nn.Linear(hidden_size_soh,3)  
        #LayerNormal
        self.lnorm =  nn.LayerNorm(d_soh_in)
        self.lnorm_soh = nn.LayerNorm(3)
        #BatchNormal
        self.bnorm_soh = nn.BatchNorm1d(d_model)

        
        
    def forward(self, x_enc,  x_dec,  
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)  
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask) 
        #enc_out size  bs*seq_len/2*d_model  64*24*512
        
        if self.args.MTLtype=='hard':       
            if self.args.decoder==False:
                if self.args.fmap==True:
                    enc_out = self.features(enc_out)

                Y_RUL = self.dropout(self.FC1(enc_out))
                Y_RUL = self.dropout(self.FC2(Y_RUL))
                Y_RUL = self.FC3(Y_RUL)
                Y_RUL = Y_RUL[:,-1,:]
                
                if self.args.loss_weight == [0,1,0]:
                    return Y_RUL 
                
                Y_HI = self.projection(enc_out)

                Y_SOH = self.lnorm(enc_out)
                Y_SOH = self.SOH_fcl(Y_SOH)  #BS*(*)*hidden_size_soh
                Y_SOH = self.SOH_fc2(Y_SOH)  #BS*(*)*3
                Y_SOH = self.lnorm_soh(Y_SOH)
                Y_SOH = F.log_softmax(Y_SOH, dim=1)  #softmax   bs*3    
                Y_SOH = Y_SOH[:,-1,:]
            
            elif self.args.decoder==True:
                dec_out = self.dec_embedding(x_dec)
                dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask) 

                if self.args.fmap==True:
                    dec_out = self.features(dec_out)
                
                Y_RUL = self.dropout(self.FC1(dec_out))
                Y_RUL = self.dropout(self.FC2(Y_RUL))
                Y_RUL = self.FC3(Y_RUL)
                Y_RUL = Y_RUL[:,-1,:]
                
                if self.args.loss_weight==[0,1,0]:
                    return Y_RUL  

                #task HI
                Y_HI = self.projection(dec_out)
                
                #Task SOH
                Y_SOH = self.lnorm(dec_out)
                Y_SOH = self.SOH_fcl(Y_SOH)  #BS*(*)*hidden_size_soh
                Y_SOH = self.SOH_fc2(Y_SOH)  #BS*(*)*3
                Y_SOH = self.lnorm_soh(Y_SOH)
                Y_SOH = F.log_softmax(Y_SOH, dim=1)  #softmax   bs*3    
                Y_SOH = Y_SOH[:,-1,:]
                
            if self.output_attention==True:
                if self.is_perception == False:
                    Y_HI = Y_HI[:,-self.pred_len:,:]
                elif self.is_perception == True:
                    Y_HI = Y_HI[:,(-self.pred_len-self.label_len):,:]
                return Y_HI, Y_SOH, Y_RUL, attns 

            elif self.output_attention==False:
                if self.is_perception == False:
                    Y_HI = Y_HI[:,-self.pred_len:,:] # [B, L, D]
                elif self.is_perception == True:
                    Y_HI = Y_HI[:,(-self.pred_len-self.label_len):,:] # [B, L, D]            
                return Y_HI, Y_SOH, Y_RUL
        
        elif self.args.MTLtype == 'hierarchy': #先soh和rul 再hi           
            dec_out = self.dec_embedding(x_dec)
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            
            Y_HI = self.projection(dec_out)
            
            if self.args.fmap==True:
                enc_out = self.features(enc_out)

            Y_RUL = self.dropout(self.FC1(enc_out))
            Y_RUL = self.dropout(self.FC2(Y_RUL))
            Y_RUL = self.FC3(Y_RUL)
            Y_RUL = Y_RUL[:,-1,:]

            Y_SOH = self.lnorm(enc_out)
            Y_SOH = self.SOH_fcl(Y_SOH)  #BS*(*)*hidden_size_soh
            Y_SOH = self.SOH_fc2(Y_SOH)  #BS*(*)*3
            Y_SOH = self.lnorm_soh(Y_SOH)
            Y_SOH = F.log_softmax(Y_SOH, dim=1)  #softmax   bs*3    
            Y_SOH = Y_SOH[:,-1,:]
      
            
            if self.output_attention==True:
                if self.is_perception == False:
                    Y_HI = Y_HI[:,-self.pred_len:,:]
                elif self.is_perception == True:
                    Y_HI = Y_HI[:,(-self.pred_len-self.label_len):,:]
                return Y_HI, Y_SOH, Y_RUL, attns 

            elif self.output_attention==False:
                if self.is_perception == False:
                    Y_HI = Y_HI[:,-self.pred_len:,:] # [B, L, D]
                elif self.is_perception == True:
                    Y_HI = Y_HI[:,(-self.pred_len-self.label_len):,:] # [B, L, D]            
                return Y_HI, Y_SOH, Y_RUL


class InformerStack(nn.Module):
    def __init__(self, OP_features, c_out, seq_len, label_len, out_len, is_perception,
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        
        if OP_features == True:
            enc_in = 20
        elif OP_features == False:
            dec_in = 14
        
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.is_perception = is_perception

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:], attns
            elif self._perception == True:
                 return dec_out[:,(-self.pred_len-self.label_len):,:], attns              
        else:
            if self.is_perception == False:
                return dec_out[:,-self.pred_len:,:] # [B, L, D]
            elif self._perception == True:
                return dec_out[:,(-self.pred_len-self.label_len):,:] # [B, L, D]

           

