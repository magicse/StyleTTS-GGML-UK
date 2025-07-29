import os
os.add_dll_directory('C:/ffmpeg/bin/')
import torch
import numpy as np
import struct
import argparse
from pathlib import Path
import yaml

def infer_model_config_from_weights(net):
    model_config = {}

    # === vocab_size (from ASR embedding layer) ===
    text_aligner = net.get('text_aligner', {})
    for name, tensor in text_aligner.items():
        if 'embedding' in name and isinstance(tensor, torch.Tensor):
            model_config['vocab_size'] = tensor.shape[0]
            print(f"[vocab_size] = {tensor.shape[0]}  ← from {name}")
            break

    # === hidden_dim — usually found in encoder or decoder ===
    # Example: text_encoder.norm.weight: [hidden_dim]
    text_encoder = net.get('text_encoder', {})
    for name, tensor in text_encoder.items():
        if 'norm.weight' in name and isinstance(tensor, torch.Tensor):
            model_config['hidden_dim'] = tensor.shape[0]
            print(f"[hidden_dim] = {tensor.shape[0]}  ← from {name}")
            break

    # === mel_channels — inferred from postnet or decoder ===
    postnet = net.get('postnet', {})
    for name, tensor in postnet.items():
        if '0.weight' in name and isinstance(tensor, torch.Tensor):
            model_config['mel_channels'] = tensor.shape[1]  # Conv: [out, in, k, k]
            print(f"[mel_channels] = {tensor.shape[1]}  ← from postnet {name}")
            break

    # === style_encoder_dim ===
    style_encoder = net.get('style_encoder', {})
    for name, tensor in style_encoder.items():
        if 'style_mlp.0.weight' in name:
            model_config['style_encoder_dim'] = tensor.shape[0]
            print(f"[style_encoder_dim] = {tensor.shape[0]}  ← from {name}")
            break

    # === text_encoder_dim (usually equals hidden_dim, but not always) ===
    for name, tensor in text_encoder.items():
        if 'proj.weight' in name:
            model_config['text_encoder_dim'] = tensor.shape[0]
            print(f"[text_encoder_dim] = {tensor.shape[0]}  ← from {name}")
            break

    # === decoder_dim ===
    decoder = net.get('decoder', {})
    for name, tensor in decoder.items():
        if 'lstm.weight_ih_l0' in name:
            model_config['decoder_dim'] = tensor.shape[1]
            print(f"[decoder_dim] = {tensor.shape[1]}  ← from {name}")
            break

    # === predictor_dim (e.g., duration_predictor.conv1.weight) ===
    duration_predictor = net.get('duration_predictor', {})
    for name, tensor in duration_predictor.items():
        if 'conv.0.weight' in name:
            model_config['predictor_dim'] = tensor.shape[1]
            print(f"[predictor_dim] = {tensor.shape[1]}  ← from {name}")
            break

    # === pitch_extractor_dim (usually in conv_block) ===
    pitch_extractor = net.get('pitch_extractor', {})
    for name, tensor in pitch_extractor.items():
        if 'conv_block.0.weight' in name:
            model_config['pitch_extractor_dim'] = tensor.shape[1]
            print(f"[pitch_extractor_dim] = {tensor.shape[1]}  ← from {name}")
            break

    # === num_speakers (if speaker embedding is present) ===
    speaker_emb = net.get('speaker_embed', {})
    for name, tensor in speaker_emb.items():
        if 'weight' in name and len(tensor.shape) == 2:
            model_config['num_speakers'] = tensor.shape[0]
            print(f"[num_speakers] = {tensor.shape[0]}  ← from {name}")
            break

    # === max_seq_len — usually set manually ===
    model_config['max_seq_len'] = 1000
    print(f"[max_seq_len] = 1000  (default, not inferred from weights)")

    return model_config


def convert_pytorch_to_ggml(model_path, config_path, output_path):
    """
    Converts a StyleTTS model from PyTorch to GGML format
    """
    print(f"Loading PyTorch model from {model_path}")
    
    # ====================================================
    from munch import Munch
    from models import load_ASR_models, load_F0_models, build_model

    config = yaml.safe_load(open(config_path))

    # Load ASR & F0 models
    text_aligner = load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
    pitch_extractor = load_F0_models(config.get('F0_path'))

    model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

    checkpoint = torch.load(model_path, map_location='cpu')
    net = checkpoint['net']
    model_config = infer_model_config_from_weights(net)
    # ====================================================
    
    print("Available model components:")
    for key in net.keys():
        print(f"  {key}")
    
    config_params = config['model_params']    
    
    # Model parameters (adjust to match your model)
    model_config = {
        'vocab_size': config_params['n_token'],
        'hidden_dim': config_params['hidden_dim'],
        'mel_channels': config_params['n_mels'],
        'max_seq_len': 1000,
        'num_speakers': 1,
        'text_encoder_dim': 512,
        'style_encoder_dim': config_params['style_dim'],
        'decoder_dim': 512,
        'predictor_dim': 512,
        'pitch_extractor_dim': 512,
    }
    
    # Open binary output file
    with open(output_path, 'wb') as f:
        # Write metadata
        f.write(struct.pack('i', model_config['vocab_size']))
        f.write(struct.pack('i', model_config['hidden_dim']))
        f.write(struct.pack('i', model_config['mel_channels']))
        f.write(struct.pack('i', model_config['max_seq_len']))
        f.write(struct.pack('i', model_config['num_speakers']))
        
        # Write configuration
        f.write(struct.pack('i', model_config['text_encoder_dim']))
        f.write(struct.pack('i', model_config['style_encoder_dim']))
        f.write(struct.pack('i', model_config['decoder_dim']))
        f.write(struct.pack('i', model_config['predictor_dim']))
        f.write(struct.pack('i', model_config['pitch_extractor_dim']))
        
        # Function to write model component
        def write_component(component_name, component_state_dict):
            print(f"\nWriting component: {component_name}")
            
            # Recursively collect all tensors from state_dict
            def collect_tensors(state_dict, prefix=""):
                tensors = {}
                for key, value in state_dict.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, torch.Tensor):
                        tensors[full_key] = value
                    elif isinstance(value, dict):
                        tensors.update(collect_tensors(value, full_key))
                return tensors
            
            tensors = collect_tensors(component_state_dict)
            num_tensors = len(tensors)
            print(f"  Found {num_tensors} tensor(s)")
            
            f.write(struct.pack('i', len(tensors)))
            
            total_params = 0
            for tensor_name, tensor in tensors.items():
                if tensor is None:
                    continue
                
                tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
                total_params += tensor_np.size
                
                print(f"  {tensor_name}: {tensor_np.shape} ({tensor_np.size} params)")
                
                name_bytes = tensor_name.encode('utf-8')
                f.write(struct.pack('i', len(name_bytes)))
                f.write(name_bytes)
                
                f.write(struct.pack('i', len(tensor_np.shape)))
                for dim in tensor_np.shape:
                    f.write(struct.pack('q', dim))  # int64_t
                
                # Write tensor data
                f.write(tensor_np.tobytes(order='F'))
            
            print(f"  Total parameters in {component_name}: {total_params}")
        
        # Write all components (exclude discriminator for inference)
        for component_name, component_state_dict in net.items():
            if component_name != 'discriminator' and component_state_dict:
                write_component(component_name, component_state_dict)
    
    print(f"\nConversion completed! Output saved to {output_path}")
    
    # Print parameter statistics
    total_params = 0
    for component_name, component_state_dict in net.items():
        if component_state_dict:
            def count_params(state_dict):
                count = 0
                for value in state_dict.values():
                    if isinstance(value, torch.Tensor):
                        count += value.numel()
                    elif isinstance(value, dict):
                        count += count_params(value)
                return count
            
            component_params = count_params(component_state_dict)
            total_params += component_params
            print(f"  {component_name}: {component_params} parameters")
    
    print(f"Total parameters converted: {total_params}")


def analyze_pytorch_model(model_path, config_path):
    """
    Analyzes the structure of a PyTorch model to understand its architecture
    """
    print(f"Analyzing PyTorch model: {model_path}")
    
