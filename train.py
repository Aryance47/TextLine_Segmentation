import os
import sys
import numpy as np
import argparse
import shutil
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
import Models
import PageLoadBatches

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set random seed for reproducibility
np.random.seed(1006)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_classes", type=int, default=1)  # Changed to 1 for binary segmentation
parser.add_argument("--input_height", type=int, default=320)
parser.add_argument("--input_width", type=int, default=320)
parser.add_argument("--epochs", type=int, default=50)  # Reduced from 250
parser.add_argument("--batch_size", type=int, default=8)  # Reduced from 16
parser.add_argument("--model_name", type=str, default="fcn8")
parser.add_argument("--optimizer_name", type=str, default="adam")  # Changed to adam
parser.add_argument("--load_weights", type=str, default='')
parser.add_argument("--train_images", type=str, default="data/preprocessed/train/")
parser.add_argument("--train_masks", type=str, default="data/preprocessed/train/")
parser.add_argument("--val_images", type=str, default="data/preprocessed/val/")
parser.add_argument("--val_masks", type=str, default="data/preprocessed/val/")

args = parser.parse_args()

# Configuration
config = {
    'n_classes': args.n_classes,
    'input_height': args.input_height,
    'input_width': args.input_width,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'model_name': args.model_name,
    'optimizer_name': args.optimizer_name,
    'load_weights': args.load_weights,
    'train_paths': {
        'images': args.train_images,
        'masks': args.train_masks
    },
    'val_paths': {
        'images': args.val_images,
        'masks': args.val_masks
    }
}

# Create directories if they don't exist
os.makedirs(config['train_paths']['images'], exist_ok=True)
os.makedirs(config['train_paths']['masks'], exist_ok=True)
os.makedirs(config['val_paths']['images'], exist_ok=True)
os.makedirs(config['val_paths']['masks'], exist_ok=True)

# Check and create validation split if needed
train_files = [f for f in os.listdir(config['train_paths']['images']) if f.endswith('_img.npy')]
val_files = [f for f in os.listdir(config['val_paths']['images']) if f.endswith('_img.npy')]

if len(val_files) == 0 and len(train_files) > 0:
    print("Creating validation split from training data...")
    train_img, val_img = train_test_split(train_files, test_size=0.2, random_state=42)
    
    for img_file in val_img:
        mask_file = img_file.replace('_img.npy', '_mask.npy')
        
        # Move image
        src = os.path.join(config['train_paths']['images'], img_file)
        dst = os.path.join(config['val_paths']['images'], img_file)
        shutil.move(src, dst)
        
        # Move mask
        src = os.path.join(config['train_paths']['masks'], mask_file)
        dst = os.path.join(config['val_paths']['masks'], mask_file)
        shutil.move(src, dst)
    
    print(f"Created validation set with {len(val_img)} samples")
    train_files = [f for f in os.listdir(config['train_paths']['images']) if f.endswith('_img.npy')]
    val_files = [f for f in os.listdir(config['val_paths']['images']) if f.endswith('_img.npy')]

# Verify we have data
assert len(train_files) > 0, f"No training files found in {config['train_paths']['images']}"
assert len(val_files) > 0, f"No validation files found in {config['val_paths']['images']}"

# Model selection (only keeping FCN models that exist)
model_fns = {
    'fcn8': Models.FCN8.FCN8,
    'fcn32': Models.FCN32.FCN32
}


# Initialize model
model = model_fns[config['model_name']](
    config['n_classes'],
    input_height=config['input_height'],
    input_width=config['input_width']
)

# Get output dimensions from model
output_shape = model.output_shape[1:3] 
output_height = output_shape[0]
output_width = output_shape[1]

# Optimizer configuration
optimizers_dict = {
    'sgd': optimizers.SGD(learning_rate=0.001, momentum=0.9),
    'adam': optimizers.Adam(learning_rate=0.0001),
    'rmsprop': optimizers.RMSprop(learning_rate=0.0001)
}
optimizer = optimizers_dict[config['optimizer_name']]

# Compile model
model.compile(
    loss='binary_crossentropy' if config['n_classes'] == 1 else 'categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Load weights if specified
if config['load_weights'] and os.path.exists(config['load_weights']):
    print(f"Loading weights from {config['load_weights']}")
    model.load_weights(config['load_weights'])

# Prepare data generators
train_gen = PageLoadBatches.imageSegmentationGenerator(
    images_path=config['train_paths']['images'],
    seg_path=config['train_paths']['masks'],
    batch_size=config['batch_size'],
    n_classes=config['n_classes'],
    input_height=config['input_height'],
    input_width=config['input_width'],
    output_height=output_height,
    output_width=output_width
)

val_gen = PageLoadBatches.imageSegmentationGenerator(
    images_path=config['val_paths']['images'],
    seg_path=config['val_paths']['masks'],
    batch_size=config['batch_size'],
    n_classes=config['n_classes'],
    input_height=config['input_height'],
    input_width=config['input_width'],
    output_height=output_height,
    output_width=output_width
)

# Callbacks
os.makedirs('weights', exist_ok=True)
callbacks = [
    ModelCheckpoint(
        filepath='weights/best_model.weights.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Calculate steps
train_steps = len(train_files) // config['batch_size']
val_steps = len(val_files) // config['batch_size']

# Training
print(f"Starting training with {len(train_files)} samples and {len(val_files)} validation samples")
print(f"Steps per epoch: {train_steps}, Validation steps: {val_steps}")

sample_x, sample_y = next(train_gen)
print(f"Input shape: {sample_x.shape}, Output shape: {sample_y.shape}")
print(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")

# Verify compatibility
assert sample_x.shape[1:] == model.input_shape[1:], "Input shape mismatch!"
assert sample_y.shape[1:] == model.output_shape[1:], "Output shape mismatch!"

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=config['epochs'],
    callbacks=callbacks
)
