import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.Diffusion_model.diffusion_encoder_nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
import torch.utils.checkpoint as activation_checkpoint
from ..layer.gnn_conv import gnn_conv
from ..act.gnn_act import gnn_act
from ..norm.gnn_norm import gnn_norm
from ..pool.gnn_pool import gnn_pool


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward


class T2TGNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, net_param,out_channels):
    super(T2TGNNEncoder, self).__init__()
    learn_norm=net_param["learn_norm"]

    n_layers=net_param["n_layers"]
    hidden_dim=net_param["hidden_dim"]
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
    self.gconv=net_param["gconv"]
    self.gconv_param=net_param["gconv_param"]

    self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    self.norm_layer=nn.BatchNorm1d(hidden_dim, affine=learn_norm)
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )
    if self.gconv=="AGNNConv":#考虑边信息更新的网络 似乎效果更好（即便边特征信息没有）
      self.edge_emb_set=True
    else:
      self.edge_emb_set = False
    if "edge_dim" in self.gconv_param:
      self.gconv_param["edge_dim"]=hidden_dim
    if self.gconv in ["GATConv" , "GATv2Conv"]:
      out_channels = hidden_dim // self.gconv_param["heads"]
      self.layers = nn.ModuleList([
        gnn_conv(gnn_name=self.gconv, in_channels=hidden_dim, out_channels=out_channels, gnn_param=self.gconv_param)
        for _ in range(n_layers)
      ])
    else:
      self.layers = nn.ModuleList([
        gnn_conv(gnn_name=self.gconv, in_channels=hidden_dim, out_channels=hidden_dim, gnn_param=self.gconv_param)
        for _ in range(n_layers)
      ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
   #可替换
    x = self.node_embed(self.pos_embed(x))
    #[节点数,嵌入维度]
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)

    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    #[节点数, 2]
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb):


    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):

      x_in, e_in = x, e


      if self.edge_emb_set:
        x, e = layer(x_in,edge_index,e_in)

      else:

        x=self.norm_layer(layer(x_in,edge_index,e_in))
        x=F.relu(x)

      x = x + time_layer(time_emb)
      x = x_in + x
      e = e_in + out_layer(e)

    return x, e

  def forward(self, x, timesteps,  edge_index=None,batch_index=None,):
    return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
#可能效果不好，待实验
class NDGNNEncoder(nn.Module):
  def __init__(self, net_param,out_channels,**kwargs):
    super(NDGNNEncoder, self).__init__()

    # 图节点原始特征信息维度

    # 节点结构信息图神经网络参数设置
    self.num_features=net_param["num_features"]
    self.prelayers_num = net_param["prelayers_num"]
    self.prelayers_dim = net_param["prelayers_dim"]
    self.prelayers_gnn = net_param["prelayers_gnn"]
    self.prelayers_gnn_param = net_param["prelayers_gnn_param"]
    self.prelayers_time_emb = net_param["prelayers_time_emb"]
    self.prelayers_act = net_param["prelayers_act"]
    self.prelayers_norm = net_param["prelayers_norm"]
    self.gstate_pool_name = net_param["gstate_pool_name"]

    self.time_embed_dim=self.prelayers_dim//2

    # 拓扑相关性图神经网络参数设置
    self.encoder_graph_state = net_param["encoder_graph_state"]
    self.enclayers_num = net_param["enclayers_num"]
    self.enclayers_dim = net_param["enclayers_dim"]
    self.enclayers_gnn = net_param["enclayers_gnn"]
    self.enclayers_gnn_param = net_param["enclayers_gnn_param"]
    self.enclayers_time_emb = net_param["enclayers_time_emb"]
    self.enclayers_act = net_param["enclayers_act"]
    self.enclayers_norm = net_param["enclayers_norm"]
    # 特征选择学习层参数设置
    self.declayers_num = net_param["declayers_num"]
    self.declayers_dim = net_param["declayers_dim"]
    self.declayers_act = net_param["declayers_act"]


    self.node_prefeatures = None
    self.node_embfeatures = None
    self.graph_state_list = None

    # Call super
    self.pre_conv_layers = torch.nn.ModuleList()
    self.pre_time_layers = torch.nn.ModuleList()
    self.pre_act = gnn_act(act_name=self.prelayers_act)
    self.pre_norms = torch.nn.ModuleList()
    self.gstate_pool = gnn_pool(pool_name=self.gstate_pool_name)
    self.encoder_conv_layers = torch.nn.ModuleList()
    self.enc_time_layers = torch.nn.ModuleList()
    self.enc_act = gnn_act(act_name=self.enclayers_act)
    self.enc_norms = torch.nn.ModuleList()
    self.decoder_layers = torch.nn.ModuleList()
    self.dec_act = gnn_act(act_name=self.declayers_act)
    #时间信息嵌入
    self.time_embed = nn.Sequential(
      linear(self.prelayers_dim, self.time_embed_dim),
      nn.ReLU(),
      linear(self.time_embed_dim, self.time_embed_dim),
    )

    self.out_channels=out_channels
    if self.prelayers_gnn=="AGNNConv":
      self.pre_edge_emb_set=True
    else:
      self.pre_edge_emb_set = False
    # 图神经网络层 学习图结构信息
    if "edge_dim" in self.prelayers_gnn_param:
      self.prelayers_gnn_param["edge_dim"]=self.prelayers_dim
    for i in range(self.prelayers_num):
      pin_channels = self.prelayers_dim if i > 0 else self.num_features
      #pin_channels = self.prelayers_dim
      plout_channels = pout_channels = self.prelayers_dim
      if self.prelayers_gnn == "GATConv":
        pout_channels=self.prelayers_dim//self.prelayers_gnn_param["head"]
      self.pre_conv_layers.append(
        gnn_conv(gnn_name=self.prelayers_gnn, in_channels=pin_channels, out_channels=pout_channels,
                 gnn_param=self.prelayers_gnn_param)

      )
      self.pre_norms.append(
        gnn_norm(norm_name=self.prelayers_norm, in_channerls=plout_channels)
      )
      if self.prelayers_time_emb :
        self.pre_time_layers.append(
          torch.nn.Linear(in_features=self.time_embed_dim, out_features=plout_channels)
        )
    if self.prelayers_num == 0:
      self.plout_channels = self.num_features
    else:
      self.plout_channels=plout_channels

    # 节点结构信息编码层——融合全局图信息
    if self.encoder_graph_state == True:
      self.encoder_W = nn.Parameter(torch.FloatTensor(plout_channels, plout_channels))
      self.decoder_W1 = nn.Parameter(torch.FloatTensor(plout_channels, 1))

    # 图神经网络层 学习节点拓扑相关性
    if self.enclayers_gnn=="AGNNConv":
      self.enc_edge_emb_set=True
    else:
      self.enc_edge_emb_set = False
    if "edge_dim" in self.enclayers_gnn_param:
      self.enclayers_gnn_param["edge_dim"] = self.plout_channels
    for i in range(self.enclayers_num):
      ein_channels = self.enclayers_dim if i > 0 else self.plout_channels

      elout_channels = eout_channels = self.enclayers_dim
      if self.enclayers_gnn in ["GATConv" , "GATv2Conv"]:
        eout_channels=self.enclayers_dim//self.enclayers_gnn_param["heads"]

      self.encoder_conv_layers.append(
        gnn_conv(gnn_name=self.enclayers_gnn, in_channels=ein_channels, out_channels=eout_channels,
                 gnn_param=self.enclayers_gnn_param)
      )
      self.enc_norms.append(
        gnn_norm(norm_name=self.enclayers_norm, in_channerls=elout_channels)
      )
      if self.enclayers_time_emb:
        self.enc_time_layers.append(
          torch.nn.Linear(in_features=self.time_embed_dim, out_features=elout_channels)
        )
    if self.enclayers_num == 0:
      self.elout_channels = self.plout_channels
    else:
      self.elout_channels=elout_channels
    embedding_size = self.elout_channels


    # 特征选择算子层 Regressor
    for i in range(self.declayers_num):
      din_channels = self.declayers_dim if i > 0 else embedding_size
      dout_channels = self.declayers_dim if i < (self.declayers_num - 1) else self.out_channels #
      self.decoder_layers.append(
        torch.nn.Linear(in_features=din_channels, out_features=dout_channels)
      )
    self.reset_parameters(self.decoder_layers)

  def reset_parameters(self, layers):
    for i in range(len(layers)):
      weight = layers[i].weight
      bias = layers[i].bias
      torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
      if bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.normal_(bias, -bound, bound)
    if self.encoder_graph_state == True:
      torch.nn.init.kaiming_normal_(self.encoder_W, a=math.sqrt(5))
      torch.nn.init.kaiming_normal_(self.decoder_W1, a=math.sqrt(5))

  def encoder(self, x, edge_index,time_emb,batch):
    # x：[N,1] 代表加噪过程中某一时间点t的节点标签值或预测值
    # edge_index:[2,edge_num]( directed edge)
    #batch :[N]
    x = x.reshape(-1, 1)
    num_nodes = x.shape[0]
    if batch is not None:
      num_graphs = int(batch[-1]) + 1
   # x = self.node_embed(self.pos_embed(x))

    # 图神经网络层 学习图结构信息 节点嵌入
    # prefeatures :x:[N,prelayers_dim]
    if self.pre_edge_emb_set:
      e = torch.zeros(edge_index.size(1), self.num_features, device=x.device)
      for i in range(self.prelayers_num):

        x, e = self.pre_conv_layers[i](x, edge_index, e)
       # print(x.shape,e.shape)
        x = self.pre_act(x) + F.relu(self.pre_time_layers[i](time_emb)) \
          if self.prelayers_time_emb \
          else self.pre_act(x)
    else:
      for i in range(self.prelayers_num):
        x = self.pre_act(self.pre_conv_layers[i](x, edge_index) )+\
            F.relu(self.pre_time_layers[i](time_emb)) \
          if self.prelayers_time_emb \
          else self.pre_act(self.pre_conv_layers[i](x, edge_index))
        # x = F.normalize(input=x,p=2.,dim=-1)
        x = self.pre_norms[i](x, batch=batch)

    # 节点结构信息编码层——融合全局图信息
    node_emb = x
    if self.encoder_graph_state == True:
      if batch is not None:
        # state_emb calculate
        # rep_global:[node_cnt,batch_size] node_cnt 是整个batch中所有图的节点数
        v = torch.ones(num_nodes).to(x.device)
        i = torch.cat((torch.arange(num_nodes).unsqueeze(0).to(x.device), batch.unsqueeze(0)), 0)
        rep_global = torch.sparse_coo_tensor(indices=i, values=v, size=(num_nodes, num_graphs))
        # [batch_size emb_size]
        state_agg = self.gstate_pool(x=node_emb, batch=batch, size=num_graphs)
        # [node_size batch_size]*[batch emb_size]=[node_zise emb_size]
        state_agg = torch.spmm(rep_global, state_agg)
        # state_emb:[node_zise emb_size]*[emb_size emb_size]=[node_zise emb_size]
        state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)
        size = node_emb.size()
        # 结合全图信息
        node_emb = node_emb.unsqueeze(2) @ state.unsqueeze(1) @ self.decoder_W1.repeat(size[0], 1).reshape(
          size[0], size[1], 1)
      else:
        state_agg = self.gstate_pool(x=node_emb, batch=None).unsqueeze(0)  # [1,emb_size]
        state = F.normalize(F.relu(state_agg @ self.encoder_W), p=2, dim=-1)
        size = node_emb.size()
        node_emb = node_emb.unsqueeze(2) @ state @ self.decoder_W1.repeat(size[0], 1).reshape(size[0], size[1], 1)
    # [batch_size*node_number enclayers_dim]
    x = node_emb
    if x.dim() != 2:
      x = node_emb.squeeze()
    self.node_prefeatures = x  # 融合全局信息的节点结构信息向量

    if not self.pre_edge_emb_set:
      e = torch.zeros(edge_index.size(1), self.plout_channels, device=x.device)
    # 学习节点拓扑相关性
    # encoder features:x[batch_size*node_number,enclayers_dim]
   # print("global",x.shape, e.shape)
    for i in range(self.enclayers_num):
      if self.enc_edge_emb_set:
       # print("ee ", e.shape)
        x,e=self.encoder_conv_layers[i](x, edge_index, e)
       # print(x.shape,e.shape)
        x = self.enc_act(x)+ F.relu(self.enc_time_layers[i](time_emb)) \
          if self.enclayers_time_emb \
          else self.enc_act(x)
      else:
        # print("x size",x.shape)
        # print("e size",e.shape)
        x = self.enc_act(self.encoder_conv_layers[i](x, edge_index,e) )+ F.relu(self.enc_time_layers[i](time_emb)) \
          if self.enclayers_time_emb \
          else self.enc_act(self.encoder_conv_layers[i](x, edge_index,e))
      x = self.enc_norms[i](x, batch=batch)
    self.node_embfeatures = x  # 融合节点拓扑相关性的节点嵌入向量
    # 特征选择学习层-回归算子
    for i in range(self.declayers_num-1):
      x = self.dec_act(self.decoder_layers[i](x))
      # x = F.normalize(input=x,p=2.,dim=-1)
    x=self.decoder_layers[-1](x)
    # [batch_size*node_number,out_channels]
    x = x.reshape(-1,self.out_channels)

    return x


  def forward(self, x, timesteps, edge_index=None ,batch_index=None):
    time_emb = self.time_embed(timestep_embedding(timesteps, self.prelayers_dim))
    # 可替换
    # [节点数, 2]
    x = self.encoder(x=x, edge_index=edge_index, time_emb=time_emb, batch=batch_index)
    return x



