import argparse
from typing import List, Tuple
from math import prod
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from utils.eval_utils import compute_var_matrix, filter_vectors_by_cond, pad_scores, score_process, get_avenue_mask, get_hr_ubnormal_mask
from utils.model_utils import processing_data
from sklearn.metrics import roc_auc_score
import os
from collections import defaultdict
from tqdm import tqdm
from utils.eval_utils import ROC
from utils.tools import get_dct_matrix, generate_pad, padding_traj
from models.advanced_gae import AdvancedGraphEncoder
from models.advanced_transformer import AdvancedTransformerEncoder, AdvancedTransformerDecoder

class TransformerAD(pl.LightningModule):

   losses = {'l1':nn.L1Loss, 'smooth_l1':nn.SmoothL1Loss, 'mse':nn.MSELoss}

   def __init__(self, args: argparse.Namespace):
       super().__init__()
       self.save_hyperparameters(args)
       # Ensure save_tensors is defined
       self.save_tensors = getattr(args, 'save_tensors', False)
       self.gt_path = getattr(args, 'gt_path', args.test_path)
       self.dropout = args.dropout
       self.num_transforms = getattr(args, 'num_transform', 1)
       self.anomaly_score_frames_shift = getattr(args, 'frames_shift', 6)
       self.anomaly_score_filter_kernel_size = getattr(args, 'filter_kernel_size', 30)
       self.dataset_name = args.dataset_choice
       self.anomaly_score_pad_size = args.pad_size
       self.n_frames = args.seg_len         # number of frames (T)
       self.n_joints = 17 if not args.headless else 14
       self.num_coords = args.num_coords      # e.g., 2 (for x,y)
       self.lr = args.opt_lr
       self.padding = args.padding
       self.cond_h_dim = args.h_dim
       self.cond_latent_dim = args.latent_dim
       self.cond_channels = args.channels
       self.cond_dropout = args.dropout
       self.n_his = args.n_his
       self.aggregation_strategy = args.aggregation_strategy
       self.noise_steps = args.noise_steps
       self.n_generated_samples = args.n_generated_samples
       self.loss_fn = self.losses[args.loss_fn](reduction='none')
       self.model_return_value = args.model_return_value
       self.idx_pad, self.zero_index = generate_pad(self.padding, self.n_his, self.n_frames-self.n_his)
    
       def get_coco_adjacency(num_joints):
          A = np.zeros((num_joints, num_joints))
          edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)
          ]
          for i, j in edges:
              A[i, j] = 1
              A[j, i] = 1  # because the graph is undirected
          # Add self-connections
          for i in range(num_joints):
              A[i, i] = 1
          return torch.tensor(A, dtype=torch.float32)

      # Prepare a fixed adjacency matrix for all samples
       self.register_buffer('adj', get_coco_adjacency(self.n_joints))
      
      # Use the AdvancedGraphEncoder from the separate file.
       self.graph_encoder = AdvancedGraphEncoder(
          in_features=self.num_coords,  # (x, y) per joint
          hidden_features=64,
          out_features=128,
          num_layers=4,  # Try deeper stacks if needed
          dropout=self.dropout,
          alpha=0.2
      )  

       # ----------------------------
       # Transformer: Now process frame-level features.

       # Define the encoder and decoder
       self.transformer_encoder = AdvancedTransformerEncoder(
            input_dim=128,
            latent_dim=256,
            ffn_dim=768,
            num_layers=6,
            num_head=8,
            dropout=self.dropout
        )

       self.transformer_decoder = AdvancedTransformerDecoder(
            latent_dim=256,
            ffn_dim=256,
            num_layers=1,
            num_head=8,
            dropout=self.dropout,
            time_embed_dim=256
        )

        # Update the projection layer to match decoder's output dimension
       self.fc_projection = nn.Linear(256, self.num_coords * self.n_joints)

   def forward(self, batch_data: List[torch.Tensor]):
        tensor_data, meta_out = self._unpack_data(batch_data)
        B, C, T, V = tensor_data.shape
        x = tensor_data.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        adj = self.adj.unsqueeze(0).repeat(B * T, 1, 1).to(x.device)
        h_out = self.graph_encoder(x, adj)
        frame_features = h_out.mean(dim=1).view(B, T, 128)
        
        # Get both encoder output and time conditioning
        enc_out, time_cond = self.transformer_encoder(frame_features)
        
        # Pass time_cond to decoder
        dec_out = self.transformer_decoder(enc_out, enc_out, time_cond)
        
        x_hat_flat = self.fc_projection(dec_out)
        x_hat = x_hat_flat.view(B, T, C, V).permute(0, 2, 1, 3)
        return x_hat, tensor_data, meta_out


   def forward_multiple(self, batch_data: List[torch.Tensor], n_samples: int = None):
        """
        Generates multiple predictions for the same input by running the forward pass
        with dropout enabled.
        
        Args:
            batch_data: Input batch.
            n_samples: Number of predictions to generate (default: self.n_generated_samples).
        
        Returns:
            predictions: List of predicted outputs.
            tensor_data, meta_out: (Taken from one forward pass; these are assumed to be the same for all runs.)
        """
        if n_samples is None:
            n_samples = self.n_generated_samples  # from config
        predictions = []
        # Enable dropout by setting the model in train mode (but disable gradient calculation)
        self.train()
        with torch.no_grad():
            for _ in range(n_samples):
                pred, tensor_data, meta_out = self.forward(batch_data)
                predictions.append(pred)
        return predictions, tensor_data, meta_out

   def forward_aggregated(self, batch_data: List[torch.Tensor], aggr_strategy: str = None, return_: str = None):
        """
        Generates multiple predictions and aggregates them into one output.
        
        Args:
            batch_data: Input batch.
            aggr_strategy: Aggregation strategy (if None, uses self.aggregation_strategy).
            return_: What to return ('poses', 'loss', or 'all').
        
        Returns:
            Aggregated output packed with additional info.
        """
        predictions, tensor_data, meta_out = self.forward_multiple(batch_data)
        selected_x, loss_of_selected_x = self._aggregation_strategy(predictions, tensor_data, aggr_strategy)
        return self._pack_out_data(selected_x, loss_of_selected_x, [tensor_data] + meta_out, return_=return_)

   def training_step(self, batch, batch_idx):
      """
      Basic reconstruction approach: MSE between input and output
      """
      x_hat, tensor_data, _ = self.forward(batch)
      loss = F.mse_loss(x_hat, tensor_data)
      self.log("loss_noise", loss)  # for consistency with the original MoCoDAD log name
      return loss


   def validation_step(self, batch, batch_idx):
      """
      Use aggregated predictions for validation if multiple predictions are generated.
      """
      if self.n_generated_samples > 1:
          output = self.forward_aggregated(batch)
      else:
          output = self.forward(batch)
      self._validation_output_list.append(output)
      return

   def on_validation_epoch_start(self):
       super().on_validation_epoch_start()
       self._validation_output_list = []


   def on_validation_epoch_end(self):
       out, gt_data, trans, meta, frames = processing_data(self._validation_output_list)
       del self._validation_output_list
       # Post-processing logic => AUC
       auc_score = self.post_processing(out, gt_data, trans, meta, frames)
       self.log('AUC', auc_score, sync_dist=True)
       return auc_score


   def test_step(self, batch, batch_idx):
        """
        Same logic as validation_step
        """
        x_hat, tensor_data, meta_out = self.forward(batch)
        
        # tensor_data has shape (B, C, T, V).
        B, C, T, V = tensor_data.shape


        # x_hat has shape (B, T, C*V) = (B, T, 34) if C=2, V=17.
        # We need to reshape x_hat to (B, 2, T, 17) so it matches tensor_data’s shape (B, C, T, V).


        # x_hat.shape -> (B, T, 34)
        # reshape => (B, T, 2, 17) => permute to (B, 2, T, 17)
        x_hat_reshaped = x_hat.reshape(B, T, C, V).permute(0, 2, 1, 3) 
        # Now x_hat_reshaped.shape = (B, 2, T, 17)


        # Then store x_hat_reshaped in the output list, so your post_processing can do (x_hat - x_true) easily.
        self._test_output_list.append([x_hat_reshaped, tensor_data] + meta_out)



   def on_test_epoch_start(self):
       super().on_test_epoch_start()
       self._test_output_list = []


   def on_test_epoch_end(self):
       """
       Reuse the same approach from MoCoDAD to compute anomaly scores + AUC
       """
       out, gt_data, trans, meta, frames = processing_data(self._test_output_list)
       del self._test_output_list
       if self.save_tensors:
           # optionally save them so you can skip direct inference next time
           pass
       auc_score = self.post_processing(out, gt_data, trans, meta, frames)
       self.log('AUC', auc_score)

   def compute_anomaly_scores_for_subset(self, out_subset, gt_data_subset, meta_subset, frames_subset):
        # Similar to code that loops over samples,
        # but here out_subset, gt_data_subset, meta_subset, and frames_subset are for a specific transformation.
        
        # If using HR-Avenue, load the mask dictionary:
        if self.dataset_name == 'HR-Avenue':
            hr_avenue_masked_clips = get_avenue_mask()  # This returns a dict: {clip_id: mask_array}

        clip_frame_scores = defaultdict(lambda: defaultdict(list))
        N_subset = len(out_subset)
        for i in range(N_subset):
            x_hat = out_subset[i]       # shape (2, T, V)
            x_true = gt_data_subset[i]  # shape (2, T, V)
            T_ = x_hat.shape[1]
            scene_idx = int(meta_subset[i, 0])
            clip_idx  = int(meta_subset[i, 1])
            if (scene_idx, clip_idx) not in self.clip_label_dict:
                continue
            clip_labels = self.clip_label_dict[(scene_idx, clip_idx)]
            frame_indices = frames_subset[i]
            valid_mask = (frame_indices >= 0) & (frame_indices < clip_labels.shape[0])
            frame_indices = frame_indices[valid_mask]
            frame_err_all = np.mean((x_hat[:, valid_mask, :] - x_true[:, valid_mask, :])**2, axis=(0,2))
            for j, f_idx in enumerate(frame_indices):
                clip_frame_scores[(scene_idx, clip_idx)][f_idx].append(frame_err_all[j])
        
        group_scores = []
        group_gt = []
        for (scene_i, clip_i), frame_dict in clip_frame_scores.items():
            if (scene_i, clip_i) not in self.clip_label_dict:
                continue
            clip_labels = self.clip_label_dict[(scene_i, clip_i)]
            clip_scores = []
            clip_labels_list = []
            for f_idx, err_list in frame_dict.items():
                # Here you can use your chosen aggregation formula
                err_arr = np.array(err_list)
                aggregated_err = np.max(err_arr)
                if 0 <= f_idx < clip_labels.shape[0]:
                    clip_scores.append(aggregated_err)
                    clip_labels_list.append(clip_labels[f_idx])
            
            # Now, if using HR-Avenue, filter clip_scores and clip_gt using the mask
            if self.dataset_name == 'HR-Avenue' and clip_i in hr_avenue_masked_clips:
                mask = np.array(hr_avenue_masked_clips[clip_i]) == 1  # assuming clip_i is the key
                # Apply mask: ensure lengths match
                if len(clip_scores) == len(mask):
                    clip_scores = np.array(clip_scores)[mask]
                    clip_labels_list = np.array(clip_labels_list)[mask]
                else:
                    print(f"Warning: Length mismatch in clip {clip_i} when applying Avenue mask.")
                    # You might choose to skip this clip or handle the mismatch appropriately.

            
            clip_scores_before = clip_scores.copy()
            clip_scores = pad_scores(clip_scores, clip_labels_list, self.anomaly_score_pad_size)
            group_scores.extend(clip_scores)
            group_gt.extend(clip_labels_list)
        return np.array(group_scores), np.array(group_gt)

   def post_processing(self, out: np.ndarray,
                    gt_data: np.ndarray,
                    trans: np.ndarray,
                    meta: np.ndarray,
                    frames: np.ndarray) -> float:

    # 1) Build a dictionary of label files
    all_label_files = [fn for fn in os.listdir(self.gt_path) if fn.endswith('.npy')]
    self.clip_label_dict = {}  # store as an attribute for reuse in the helper function
    for fname in sorted(all_label_files):
        base = os.path.splitext(fname)[0]  # e.g. "0_1"
        scene_str, clip_str = base.split('_')
        scene_i = int(scene_str)
        clip_i = int(clip_str)
        label_arr = np.load(os.path.join(self.gt_path, fname))
        self.clip_label_dict[(scene_i, clip_i)] = label_arr

    # 2) Group sample indices by transformation
    transform_groups = {}
    N = len(trans)
    for i in range(N):
        t_idx = int(trans[i])
        if t_idx not in transform_groups:
            transform_groups[t_idx] = []
        transform_groups[t_idx].append(i)
    
    # 3) For each transformation, compute anomaly scores
    model_scores_transf = {}
    dataset_gt_transf = {}
    for t in range(self.num_transforms):
        indices = transform_groups.get(t, [])
        if len(indices) == 0:
            continue
        out_subset = out[indices]
        gt_data_subset = gt_data[indices]
        meta_subset = meta[indices]
        frames_subset = frames[indices]
        scores_t, gt_t = self.compute_anomaly_scores_for_subset(out_subset, gt_data_subset, meta_subset, frames_subset)
        model_scores_transf[t] = scores_t
        dataset_gt_transf[t] = gt_t

    # 4) Aggregate anomaly scores across transformations
    if len(model_scores_transf) == 0:
        print("No transformation outputs available.")
        return float('nan')
    all_scores = np.max(np.stack(list(model_scores_transf.values()), axis=0), axis=0)
    all_gt = dataset_gt_transf[0]

    # Apply Gaussian smoothing and score processing
    all_scores = score_process(all_scores, self.anomaly_score_frames_shift, self.anomaly_score_filter_kernel_size)
  
     # Ensure directory exists
    os.makedirs("roc_plots", exist_ok=True)
    
    # Generate ROC plot with dataset name
    roc_path = os.path.join("roc_plots", 
        f"{self.dataset_name}_roc_epoch_{self.current_epoch}.png")
    
    _, auc_score = ROC(all_gt, all_scores, 
                      dataset_name=self.dataset_name.upper(),
                      path=roc_path)
    
    # Log image with clearer name
    if self.logger:
        import matplotlib.image as mpimg
        image = mpimg.imread(roc_path)
        image_tensor = torch.tensor(image).permute(2, 0, 1)
        self.logger.experiment.add_image(
            f"ROC/{self.dataset_name}",
            image_tensor, 
            self.current_epoch
        )
    
    return auc_score


   def configure_optimizers(self):
       return torch.optim.Adam(self.parameters(), lr=self.lr)


   def _unpack_data(self, batch):
        if isinstance(batch, (list, tuple)):
            # If it has at least 4 elements, unpack normally.
            if len(batch) >= 4:
                tensor_data = batch[0].to(self.device)
                transformation_idx = batch[1]
                metadata = batch[2]
                actual_frames = batch[3]
                meta_out = [transformation_idx, metadata, actual_frames]
            else:
                # Otherwise, assume the list only contains the tensor.
                tensor_data = batch[0].to(self.device)
                meta_out = []
        else:
            tensor_data = batch.to(self.device)
            meta_out = []
        return tensor_data, meta_out


   def _pack_out_data(self, selected_x:torch.Tensor, loss_of_selected_x:torch.Tensor, additional_out:List[torch.Tensor], return_:str) -> List[torch.Tensor]:
            """
            Packs the output data according to the return_ argument.

            Args:
                selected_x (torch.Tensor): generated samples selected among the others according to the aggregation strategy
                loss_of_selected_x (torch.Tensor): loss of the selected samples
                additional_out (List[torch.Tensor]): additional output data (ground truth, meta-data, etc.)
                return_ (str): return strategy. Can be 'pose', 'loss', 'all'

            Raises:
                ValueError: if return_ is None and self.model_return_value is None

            Returns:
                List[torch.Tensor]: output data
            """
            
            if return_ is None:
                if self.model_return_value is None:
                    raise ValueError('Either return_ or self.model_return_value must be set')
                else:
                    return_ = self.model_return_value

            if return_ == 'poses':
                out = [selected_x]
            elif return_ == 'loss':
                out = [loss_of_selected_x]
            elif return_ == 'all':
                out = [loss_of_selected_x, selected_x]
                
            return out + additional_out

   def _aggregation_strategy(self, generated_xs:List[torch.Tensor], input_sequence:torch.Tensor, aggr_strategy:str) -> Tuple[torch.Tensor]:
        """
        Aggregates the generated samples and returns the selected one and its reconstruction error.
        Strategies:
            - all: returns all the generated samples
            - random: returns a random sample
            - best: returns the sample with the lowest reconstruction loss
            - worst: returns the sample with the highest reconstruction loss
            - mean: returns the mean of the losses of the generated samples
            - median: returns the median of the losses of the generated samples
            - mean_pose: returns the mean of the generated samples
            - median_pose: returns the median of the generated samples

        Args:
            generated_xs (List[torch.Tensor]): list of generated samples
            input_sequence (torch.Tensor): ground truth sequence
            aggr_strategy (str): aggregation strategy

        Raises:
            ValueError: if the aggregation strategy is not valid

        Returns:
            Tuple[torch.Tensor]: selected sample and its reconstruction error
        """

        aggr_strategy = self.aggregation_strategy if aggr_strategy is None else aggr_strategy 
        if aggr_strategy == 'random':
            return generated_xs[np.random.randint(len(generated_xs))]
        
        B, repr_shape = input_sequence.shape[0], input_sequence.shape[1:]
        compute_loss = lambda x: torch.mean(self.loss_fn(x, input_sequence).reshape(-1, prod(repr_shape)), dim=-1)
        losses = [compute_loss(x) for x in generated_xs]

        if aggr_strategy == 'all':
            dims_idxs = list(range(2, len(repr_shape)+2))
            dims_idxs = [1,0] + dims_idxs
            selected_x = torch.stack(generated_xs).permute(*dims_idxs)
            loss_of_selected_x = torch.stack(losses).permute(1,0)
        elif aggr_strategy == 'mean':
            selected_x = None
            loss_of_selected_x = torch.mean(torch.stack(losses), dim=0)
        elif aggr_strategy == 'mean_pose':
            selected_x = torch.mean(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'median':
            loss_of_selected_x, _ = torch.median(torch.stack(losses), dim=0)
            selected_x = None
        elif aggr_strategy == 'median_pose':
            selected_x, _ = torch.median(torch.stack(generated_xs), dim=0)
            loss_of_selected_x = compute_loss(selected_x)
        elif aggr_strategy == 'best' or aggr_strategy == 'worst':
            strategy = (lambda x,y: x < y) if aggr_strategy == 'best' else (lambda x,y: x > y)
            loss_of_selected_x = torch.full((B,), fill_value=(1e10 if aggr_strategy == 'best' else -1.), device=self.device)
            selected_x = torch.zeros((B, *repr_shape)).to(self.device)

            for i in range(len(generated_xs)):
                mask = strategy(losses[i], loss_of_selected_x)
                loss_of_selected_x[mask] = losses[i][mask]
                selected_x[mask] = generated_xs[i][mask]
        elif 'quantile' in aggr_strategy:
            q = float(aggr_strategy.split(':')[-1])
            loss_of_selected_x = torch.quantile(torch.stack(losses), q, dim=0)
            selected_x = None
        else:
            raise ValueError(f'Unknown aggregation strategy {aggr_strategy}')
        
        return selected_x, loss_of_selected_x