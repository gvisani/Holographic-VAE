
'''

NOT TESTED YET

'''


import sys, os
import numpy as np
import torch
import e3nn
from e3nn import o3
from holographic_vae import nn
from holographic_vae.so3.functional import make_vec
from holographic_vae.utils.loss_functions import *

from torch import Tensor
from typing import *



class H_VAE(torch.nn.Module):

    def load_hparams(self, hparams):
        self.latent_dim = hparams['latent_dim'] # int
        self.net_lmax = hparams['net_lmax'] # int

        self.ch_size_list = hparams['ch_size_list'] # List[int] ; as they showld appear in the encoder
        self.lmax_list = hparams['lmax_list'] # Optional[List[int]] ; as they should appear in the encoder
        self.ls_nonlin_rule_list = hparams['ls_nonlin_rule_list'] # List[str] ; as they should appear in the encoder
        self.ch_nonlin_rule_list = hparams['ch_nonlin_rule_list'] # List[str] ; as they should appear in the encoder
        self.do_initial_linear_projection = hparams['do_initial_linear_projection'] # bool
        self.ch_initial_linear_projection = hparams['ch_initial_linear_projection'] # int
        self.bottleneck_hidden_dims = hparams['bottleneck_hidden_dims'] # Optional[List[int]], as they should appear in the encoder
        self.use_additive_skip_connections = hparams['use_additive_skip_connections'] # bool
        self.use_batch_norm = hparams['use_batch_norm'] # bool
        self.weights_initializer = hparams['weights_initializer'] # Optional[str]
        self.norm_type = hparams['norm_type'] # Optional[str] ; None, layer, signal
        self.normalization = hparams['normalization'] # str ; norm, component
        self.norm_balanced = hparams['norm_balanced'] # bool ; only for signal norm
        self.norm_affine = hparams['norm_affine'] # Optional[Union[str, bool]] ; None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
        self.norm_nonlinearity = hparams['norm_nonlinearity'] # Optional[str] ; None (identity), identity, relu, swish, sigmoid -> only for layer_norm
        self.norm_location = hparams['norm_location'] # str ; first, between, last
        self.linearity_first = hparams['linearity_first'] # bool ; currently only works with this being false
        self.filter_symmetric = hparams['filter_symmetric'] # bool ; whether to exclude duplicate pairs of l's from the tensor product nonlinearity
        self.x_rec_loss_fn = hparams['x_rec_loss_fn'] # str ; mse, mse_normalized, cosine
        self.do_final_signal_norm = hparams['do_final_signal_norm'] # bool
        self.learn_frame = hparams['learn_frame'] # bool
        self.is_vae = hparams['is_vae'] # bool

        self.input_normalizing_constant = torch.tensor(hparams['input_normalizing_constant'], requires_grad=False) if hparams['input_normalizing_constant'] is not None else None

        assert self.lmax_list is None or len(self.ch_size_list) == len(self.lmax_list)
        assert len(self.ch_size_list) == len(self.ls_nonlin_rule_list)
        assert len(self.ch_size_list) == len(self.ch_nonlin_rule_list)
        self.n_cg_blocks = len(self.ch_size_list)

    def __init__(self,
                 irreps_in: o3.Irreps,
                 w3j_matrices: Dict[int, Tensor],
                 hparams: Dict,
                 device: str,
                 normalize_input_at_runtime: bool = False
                 ):
        super().__init__()

        self.irreps_in = irreps_in
        self.load_hparams(hparams)
        self.device = device
        self.normalize_input_at_runtime = normalize_input_at_runtime


        if self.do_initial_linear_projection:
            print(irreps_in.dim, irreps_in)
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(irreps_in.ls), 1)).sort().irreps.simplify()
            self.initial_linear_projection = nn.SO3_linearity(irreps_in, initial_irreps)
            print(initial_irreps.dim, initial_irreps)
        else:
            print(irreps_in.dim, irreps_in)
            initial_irreps = irreps_in
        
        # prepare lmaxs for both encoder and decoder blocks
        lmaxs = [min(2**i, self.net_lmax) for i in range(self.n_cg_blocks)] + [max(initial_irreps.ls)]
        lmaxs_encoder_upper_bound = lmaxs[:-1][::-1] # exclude data irreps and reverse so it's decreasing
        lmaxs_decoder_upper_bound = lmaxs[1:] # exclude latent space irreps
        if self.lmax_list is not None:
            lmaxs_encoder = self.lmax_list
            lmaxs_decoder = self.lmax_list[::-1][1:] + [max(initial_irreps.ls)]

            # the provided lmaxs must be such that they are not above the maximum
            # allowed by the bottleneck that goes down to lmax=1
            assert np.all(np.array(lmaxs_encoder_upper_bound) >= np.array(lmaxs_encoder))
            assert np.all(np.array(lmaxs_decoder_upper_bound) >= np.array(lmaxs_decoder))
        else:
            lmaxs_encoder = lmaxs_encoder_upper_bound
            lmaxs_decoder = lmaxs_decoder_upper_bound

        ## encoder - cg
        prev_irreps = initial_irreps
        encoder_cg_blocks = []
        for i in range(self.n_cg_blocks):
            temp_irreps_hidden = (self.ch_size_list[i]*o3.Irreps.spherical_harmonics(lmaxs_encoder[i], 1)).sort().irreps.simplify()
            encoder_cg_blocks.append(nn.CGBlock(prev_irreps,
                                                temp_irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=self.linearity_first,
                                                filter_symmetric=self.filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=self.ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=self.ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=self.norm_balanced,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=self.weights_initializer,
                                                init_scale=1.0))


            prev_irreps = encoder_cg_blocks[-1].irreps_out
            print(prev_irreps.dim, prev_irreps)

        self.encoder_cg_blocks = torch.nn.ModuleList(encoder_cg_blocks)

        final_encoder_invariants = [mul for (mul, _) in prev_irreps][0] # number of channels for l = 0
        final_encoder_l1s = [mul for (mul, _) in prev_irreps][1] # number of channels for l = 1

        prev_dim = final_encoder_invariants
        if self.bottleneck_hidden_dims is not None and len(self.bottleneck_hidden_dims) > 0:
            encoder_bottleneck = []
            for hidden_dim in self.bottleneck_hidden_dims:
                encoder_bottleneck.append(torch.nn.Linear(prev_dim, hidden_dim))
                encoder_bottleneck.append(torch.nn.ReLU())
                prev_dim = hidden_dim
            self.encoder_bottleneck = torch.nn.Sequential(*encoder_bottleneck)
        else:
            self.encoder_bottleneck = torch.nn.Identity() # for modularity purposes

        self.encoder_mean = torch.nn.Linear(prev_dim, self.latent_dim)
        if self.is_vae:
            self.encoder_log_var = torch.nn.Linear(prev_dim, self.latent_dim)

        # component that learns the frame
        if self.learn_frame:
            # take l=1 vectors (extract multiplicities) of last block and learn two l=1 vectors (x and pseudo-y direction)
            frame_learner_irreps_in = o3.Irreps('%dx1e' % final_encoder_l1s)
            frame_learner_irreps_out = o3.Irreps('2x1e')
            self.frame_learner = nn.SO3_linearity(frame_learner_irreps_in, frame_learner_irreps_out)
        
        latent_irreps = o3.Irreps('%dx0e+3x1e' % (self.latent_dim))
        print(latent_irreps.dim, latent_irreps)

        prev_dim = self.latent_dim
        if self.bottleneck_hidden_dims is not None and len(self.bottleneck_hidden_dims) > 0:
            bottleneck_hidden_dims = self.bottleneck_hidden_dims[::-1] + [final_encoder_invariants]
            decoder_bottleneck = []
            for i, hidden_dim in enumerate(bottleneck_hidden_dims):
                decoder_bottleneck.append(torch.nn.Linear(prev_dim, hidden_dim))
                if i > 0: # only linear projection in first layer, to be symmetric with encoder
                    decoder_bottleneck.append(torch.nn.ReLU())
                prev_dim = hidden_dim
            self.decoder_bottleneck = torch.nn.Sequential(*decoder_bottleneck)
        else:
            self.decoder_bottleneck = torch.nn.Linear(self.latent_dim, final_encoder_invariants)

        l1_frame_irreps = o3.Irreps('3x1e')
        self.first_decoder_projection = nn.SO3_linearity(l1_frame_irreps, o3.Irreps('%dx1e' % final_encoder_l1s)) # project back to space of irreps at the end of the encoder (l1 only)
        print(prev_irreps.dim, prev_irreps)

        ## decoder - cg
        decoder_cg_blocks = []
        
        ch_size_list = self.ch_size_list[::-1][1:] # reverse and exclude first channels because that's the input to the decoder
        ls_nonlin_rule_list = self.ls_nonlin_rule_list[::-1]
        ch_nonlin_rule_list = self.ch_nonlin_rule_list[::-1]
        for i in range(self.n_cg_blocks):
            if i == self.n_cg_blocks - 1:
                if self.do_initial_linear_projection:
                    temp_irreps_hidden = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(lmaxs_decoder[i], 1)).sort().irreps.simplify()
                else:
                    temp_irreps_hidden = irreps_in
            else:
                temp_irreps_hidden = (ch_size_list[i]*o3.Irreps.spherical_harmonics(lmaxs_decoder[i], 1)).sort().irreps.simplify()
            decoder_cg_blocks.append(nn.CGBlock(prev_irreps,
                                                temp_irreps_hidden,
                                                w3j_matrices,
                                                linearity_first=self.linearity_first,
                                                filter_symmetric=self.filter_symmetric,
                                                use_batch_norm=self.use_batch_norm,
                                                ls_nonlin_rule=ls_nonlin_rule_list[i], # full, elementwise, efficient
                                                ch_nonlin_rule=ch_nonlin_rule_list[i], # full, elementwise
                                                norm_type=self.norm_type, # None, layer, signal
                                                normalization=self.normalization, # norm, component -> only if norm_type is not none
                                                norm_balanced=self.norm_balanced,
                                                norm_affine=self.norm_affine, # None, {True, False} -> for layer_norm, {unique, per_l, per_feature} -> for signal_norm
                                                norm_nonlinearity=self.norm_nonlinearity, # None (identity), identity, relu, swish, sigmoid -> only for layer_norm
                                                norm_location=self.norm_location, # first, between, last
                                                weights_initializer=self.weights_initializer,
                                                init_scale=1.0))

            prev_irreps = temp_irreps_hidden
            print(prev_irreps.dim, prev_irreps)

        self.decoder_cg_blocks = torch.nn.ModuleList(decoder_cg_blocks)

        if self.do_initial_linear_projection: # final linear projection
            initial_irreps = (self.ch_initial_linear_projection*o3.Irreps.spherical_harmonics(max(irreps_in.ls), 1)).sort().irreps.simplify()
            self.final_linear_projection = nn.SO3_linearity(initial_irreps, irreps_in)
            print(irreps_in.dim, irreps_in)

        if self.do_final_signal_norm:
            self.final_signal_norm = torch.nn.Sequential(nn.signal_norm(irreps_in, normalization='component', affine=None))

        ## setup reconstruction loss functions
        self.signal_rec_loss = eval(NAME_TO_LOSS_FN[self.x_rec_loss_fn])(irreps_in, self.device)
    
    def encode(self, x: Dict[int, Tensor]):

        # print('----- START ENCODER -----', file=sys.stderr)

        # normalize input data if desired
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            # print('Normalizing input data by constant ', self.input_normalizing_constant, file=sys.stderr)
            for l in x:
                x[l] = x[l] / self.input_normalizing_constant

        if self.do_initial_linear_projection:
            h = self.initial_linear_projection(x)
        else:
            h = x
        
        for i, block in enumerate(self.encoder_cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp
            
            if self.learn_frame and i == len(self.encoder_cg_blocks) - 1:
                last_l1_values = {1: h[1]}
        
        encoder_invariants = self.encoder_bottleneck(h[0].squeeze(-1))
        z_mean = self.encoder_mean(encoder_invariants)
        if self.is_vae:
            z_log_var = self.encoder_log_var(encoder_invariants)
        else:
            z_log_var = torch.zeros_like(z_mean) # placeholder value that won't throw errors
        

        if self.learn_frame:

            # print('Is learned frame before Linear nan: ', torch.any(torch.isnan(last_l1_values[1])).item(), file=sys.stderr)
            # print('Is learned frame before Linear inf: ', torch.any(torch.isinf(last_l1_values[1])).item(), file=sys.stderr)

            before_orthonorm = self.frame_learner(last_l1_values)[1]

            # print('Is learned frame before Orthonorm nan: ', torch.any(torch.isnan(last_l1_values[1])).item(), file=sys.stderr)
            # print('Is learned frame before Orthonorm inf: ', torch.any(torch.isinf(last_l1_values[1])).item(), file=sys.stderr)

            learned_frame = self.orthonormalize_frame(before_orthonorm)
        else:
            learned_frame = None

        # print('----- END ENCODER -----', file=sys.stderr)

        return (z_mean, z_log_var), learned_frame
    
    def decode(self, z: Tensor, frame: Tensor):

        # print('----- START DECODER -----', file=sys.stderr)

        h = self.first_decoder_projection({1: frame})
        h[0] = self.decoder_bottleneck(z).unsqueeze(-1)
        for i, block in enumerate(self.decoder_cg_blocks):
            h_temp = block(h)
            if self.use_additive_skip_connections:
                for l in h:
                    if l in h_temp:
                        if h[l].shape[1] == h_temp[l].shape[1]: # the shape at index 1 is the channels' dimension
                            h_temp[l] += h[l]
                        elif h[l].shape[1] > h_temp[l].shape[1]:
                            h_temp[l] += h[l][:, : h_temp[l].shape[1], :] # subsample first channels
                        else: # h[l].shape[1] < h_temp[l].shape[1]
                            h_temp[l] += torch.nn.functional.pad(h[l], (0, 0, 0, h_temp[l].shape[1] - h[l].shape[1])) # zero pad the channels' dimension
            h = h_temp

        if self.do_initial_linear_projection:
            x_reconst = self.final_linear_projection(h)
        else:
            x_reconst = h

        if self.do_final_signal_norm:
            x_reconst = self.final_signal_norm(x_reconst)
        
        # print('----- END DECODER -----', file=sys.stderr)

        return x_reconst

    def forward(self, x: Dict[int, Tensor], x_vec: Optional[Tensor] = None, frame: Optional[Tensor] = None):
        '''
        Note: this function is independent of the choice of probability distribution for the latent space,
              and of the choice of encoder and decoder. Only the inputs and outputs must be respected
        '''

        distribution_params, learned_frame = self.encode(x)
        if self.is_vae:
            z = self.reparameterization(*distribution_params)
        else:
            z = distribution_params[0]

        if self.learn_frame:
            frame = learned_frame

        x_reconst = self.decode(z, frame)

        # gather loss values
        x_reconst_vec = make_vec(x_reconst)

        if x_vec is None:
            x_vec = make_vec(x) # NOTE: doing this is sub-optimal! better to just provide it
        
        if self.normalize_input_at_runtime and self.input_normalizing_constant is not None:
            x_vec = x_vec / self.input_normalizing_constant

        x_reconst_loss = self.signal_rec_loss(x_reconst_vec, x_vec)

        if self.is_vae:
            kl_divergence = self.kl_divergence(*distribution_params) / self.latent_dim  # KLD is summed over each latent variable, so it's better to divide it by the latent dim
                                                                                        # to get a value that is independent (or less dependent) of the latent dim size
        else:
            kl_divergence = torch.tensor(0.0) # placeholder value that won't throw errors

        return x_reconst_loss, kl_divergence, x_reconst, distribution_params
    
    def reparameterization(self, mean: Tensor, log_var: Tensor):

        # isotropic gaussian latent space
        stddev = torch.exp(0.5 * log_var) # takes exponential function (log var -> stddev)
        epsilon = torch.randn_like(stddev).to(self.device)        # sampling epsilon        
        z = mean + stddev*epsilon                          # reparameterization trick

        return z
    
    def kl_divergence(self, z_mean: Tensor, z_log_var: Tensor):
        # isotropic normal prior on the latent space
        return torch.mean(- 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1))
    
    def orthonormalize_frame(self, x_psy_N6, eps=0.0):
        '''
        Gram-Schmidt process
        
        y = psy - (<x, psy> / <x, x>) x
        z = x \cross y

        x = x / ||x||
        y = y / ||y||
        z = z / ||z||


        alternative:
        xp = x / ||x||
        yp = psy - <xp, psy> xp / || psy - <x, psy> xp ||
        zp = xp \cross yp

        '''
        
        x, psy = x_psy_N6[:, 0, :], x_psy_N6[:, 1, :]
        

        x_dot_psy = torch.sum(torch.mul(x, psy), dim=1).view(-1, 1)
        x_dot_x = torch.sum(torch.mul(x, x), dim=1).view(-1, 1)
        y = psy - (x_dot_psy / (x_dot_x + eps)) * x
        z = torch.cross(x, y, dim=1)

        # print(torch.min(torch.sum(torch.mul(x, x), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(x, psy), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(psy, psy), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(y, y), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(z, z), dim=1)).item(), file=sys.stderr)
        
        x = x / (torch.sqrt(torch.sum(torch.mul(x, x), dim=1).view(-1, 1)) + eps)
        y = y / (torch.sqrt(torch.sum(torch.mul(y, y), dim=1).view(-1, 1)) + eps)
        z = z / (torch.sqrt(torch.sum(torch.mul(z, z), dim=1).view(-1, 1)) + eps)


        # print(torch.min(torch.sum(torch.mul(x, x), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(psy, psy), dim=1)).item(), file=sys.stderr)

        # x = x / (torch.sqrt(torch.sum(torch.mul(x, x), dim=1)).view(-1, 1) + eps)
        # x_dot_psy = torch.sum(torch.mul(x, psy), dim=1).view(-1, 1)
        # y = psy - x_dot_psy * x
        # y = y / (torch.sqrt(torch.sum(torch.mul(y, y), dim=1)).view(-1, 1) + eps)
        # z = torch.cross(x, y, dim=1)

        # print(torch.min(torch.sum(torch.mul(x, x), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(y, y), dim=1)).item(), file=sys.stderr)
        # print(torch.min(torch.sum(torch.mul(z, z), dim=1)).item(), file=sys.stderr)


        xyz = torch.stack([x, y, z], dim=1)


        
        return xyz
