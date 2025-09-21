"""
Model loading utilities with compatibility fixes for different TensorFlow versions
"""
import os
import json
import h5py
import tempfile
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
import tensorflow as tf


def safe_load_model(model_path, compile_model=False):
    """
    Safely load a Keras model with compatibility fixes for version differences.
    
    Args:
        model_path (str): Path to the .h5 model file
        compile_model (bool): Whether to compile the model after loading
        
    Returns:
        tensorflow.keras.Model: Loaded model
        
    Raises:
        Exception: If model cannot be loaded with any method
    """
    print(f"Attempting to load model: {model_path}")
    
    # Method 1: Try normal loading first
    try:
        model = load_model(model_path, compile=compile_model)
        print("SUCCESS: Model loaded successfully with standard method")
        return model
    except Exception as e:
        print(f"FAILED: Standard loading failed: {str(e)[:100]}...")
        
        # Check if it's the specific groups parameter error
        if "groups" in str(e) and "DepthwiseConv2D" in str(e):
            print("INFO: Detected DepthwiseConv2D 'groups' parameter issue")
            return _load_model_with_groups_fix(model_path, compile_model)
        else:
            raise e


def _load_model_with_groups_fix(model_path, compile_model=False):
    """
    Load model by removing the problematic 'groups' parameter from DepthwiseConv2D layers.
    """
    # Method 2: Try with custom DepthwiseConv2D
    try:
        def custom_depthwise_conv2d(*args, **kwargs):
            # Remove 'groups' parameter if present
            kwargs.pop('groups', None)
            return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)
        
        with CustomObjectScope({'DepthwiseConv2D': custom_depthwise_conv2d}):
            model = load_model(model_path, compile=compile_model)
            print("SUCCESS: Model loaded with custom DepthwiseConv2D")
            return model
    except Exception as e:
        print(f"FAILED: Custom object scope failed: {str(e)[:100]}...")
    
    # Method 3: Fix the model file directly
    try:
        return _fix_model_config_and_load(model_path, compile_model)
    except Exception as e:
        print(f"FAILED: Config fix method failed: {str(e)[:100]}...")
        raise Exception(f"All model loading methods failed for {model_path}")


def _fix_model_config_and_load(model_path, compile_model=False):
    """
    Fix the model configuration by removing 'groups' parameter and reload.
    """
    # Create temporary copy of the model file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
        shutil.copy2(model_path, tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Fix the model configuration
        with h5py.File(temp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                
                config = json.loads(config_str)
                
                # Remove 'groups' parameter from all DepthwiseConv2D layers
                groups_removed = _remove_groups_parameter(config)
                
                if groups_removed > 0:
                    print(f"INFO: Removed 'groups' parameter from {groups_removed} layers")
                    # Save fixed configuration
                    fixed_config_str = json.dumps(config)
                    f.attrs['model_config'] = fixed_config_str.encode('utf-8')
        
        # Load the fixed model
        model = load_model(temp_path, compile=compile_model)
        print("SUCCESS: Model loaded with config fix")
        return model
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _remove_groups_parameter(obj, removed_count=0):
    """
    Recursively remove 'groups' parameter from DepthwiseConv2D layer configurations.
    """
    if isinstance(obj, dict):
        # Check if this is a DepthwiseConv2D layer config
        if (obj.get('class_name') == 'DepthwiseConv2D' and 
            'config' in obj and 
            'groups' in obj['config']):
            del obj['config']['groups']
            removed_count += 1
        
        # Recursively process all dict values
        for value in obj.values():
            removed_count = _remove_groups_parameter(value, removed_count)
    
    elif isinstance(obj, list):
        # Process all list items
        for item in obj:
            removed_count = _remove_groups_parameter(item, removed_count)
    
    return removed_count
