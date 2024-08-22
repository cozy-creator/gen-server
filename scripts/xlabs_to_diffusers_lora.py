
import re



def convert_lora_keys(old_state_dict):
    new_state_dict = {}
    
    for old_key, value in old_state_dict.items():
        # Handle double_blocks
        if old_key.startswith('double_blocks'):
            block_num = re.search(r'double_blocks\.(\d+)', old_key).group(1)
            new_key = f'transformer.transformer_blocks.{block_num}'
            
            if 'processor.proj_lora1' in old_key:
                new_key += '.attn.to_out.0'
            elif 'processor.proj_lora2' in old_key:
                new_key += '.attn.to_add_out'
            elif 'processor.qkv_lora1' in old_key:
                new_key += '.attn'
                if 'down' in old_key:
                    for proj in ['add_q_proj', 'add_k_proj', 'add_v_proj']:
                        proj_key = f'{new_key}.{proj}.lora_A.weight'
                        new_state_dict[proj_key] = value
                elif 'up' in old_key:
                    for proj in ['add_q_proj', 'add_k_proj', 'add_v_proj']:
                        proj_key = f'{new_key}.{proj}.lora_B.weight'
                        new_state_dict[proj_key] = value
                continue
            elif 'processor.qkv_lora2' in old_key:
                new_key += '.attn'
                if 'down' in old_key:
                    for proj in ['to_q', 'to_k', 'to_v']:
                        proj_key = f'{new_key}.{proj}.lora_A.weight'
                        new_state_dict[proj_key] = value
                elif 'up' in old_key:
                    for proj in ['to_q', 'to_k', 'to_v']:
                        proj_key = f'{new_key}.{proj}.lora_B.weight'
                        new_state_dict[proj_key] = value
                continue
            
            if 'down' in old_key:
                new_key += '.lora_A.weight'
            elif 'up' in old_key:
                new_key += '.lora_B.weight'
            
        # Handle single_blocks
        elif old_key.startswith('single_blocks'):
            block_num = re.search(r'single_blocks\.(\d+)', old_key).group(1)
            new_key = f'transformer.single_transformer_blocks.{block_num}'
            
            if 'proj_lora1' in old_key or 'proj_lora2' in old_key:
                new_key += '.proj_out'
            elif 'qkv_lora1' in old_key or 'qkv_lora2' in old_key:
                new_key += '.norm.linear'
            
            if 'down' in old_key:
                new_key += '.lora_A.weight'
            elif 'up' in old_key:
                new_key += '.lora_B.weight'
        
        else:
            # Handle other potential key patterns here
            new_key = old_key
        
        new_state_dict[new_key] = value
    
    return new_state_dict