import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
from typing import Tuple, List, Optional

# Set random seed for reproducibility
np.random.seed(42)

# CIFAR-10 Classes
classes = ["airplane", "frog", "truck", "horse", "deer", 
           "automobile", "bird", "ship", "cat", "dog"]
label_integer = {classes[i]: i for i in range(len(classes))}

class CIFAR10Loader:
    """Load and preprocess CIFAR-10 data from PNG files and CSV"""
    
    def __init__(self):
        self.classes = classes
        self.label_integer = label_integer
    
    def load_all_data(self, data_path: str = "/home/yagati/Downloads/cifar-10") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load all CIFAR-10 data from PNG files and CSV"""
        print(f"Loading CIFAR-10 dataset from {data_path}...")
        
        # Define file paths
        csv_path = os.path.join(data_path, "trainLabels.csv")
        train_dir = os.path.join(data_path, "train")
        
        print(f"CSV path: {csv_path}")
        print(f"Train directory: {train_dir}")
        
        # Check if paths exist
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        
        # Load all labels from CSV
        print("Loading labels from CSV...")
        raw_labels = np.genfromtxt(csv_path, dtype=str, delimiter=',', skip_header=1)
        all_labels = np.array([self.label_integer[label] for label in raw_labels[:, 1]])
        
        # Load all images
        all_images = []
        total_images = len(all_labels)
        
        print(f"Loading {total_images} images from {train_dir}...")
        for i in range(total_images):
            file_path = os.path.join(train_dir, f"{i + 1}.png")
            if os.path.exists(file_path):
                image = img.imread(file_path)
                all_images.append(image)
            else:
                print(f"Warning: {file_path} not found")
                # Create dummy image if file not found
                dummy_image = np.random.rand(32, 32, 3).astype(np.float32)
                all_images.append(dummy_image)
            
            # Show progress
            if (i + 1) % 1000 == 0:
                print(f"Loaded {i + 1}/{total_images} images...")
        
        all_images = np.array(all_images)
        print(f"Successfully loaded {len(all_images)} images with shape {all_images.shape}")
        
        # Convert from NHWC to NCHW format (channels first)
        all_images = all_images.transpose(0, 3, 1, 2)
        
        # Split into train and test (80-20 split)
        split_idx = int(0.8 * total_images)
        X_train = all_images[:split_idx]
        y_train = all_labels[:split_idx]
        X_test = all_images[split_idx:]
        y_test = all_labels[split_idx:]
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_data(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess CIFAR-10 data"""
        print("Preprocessing data...")
        
        # Convert to float32 and normalize to [0, 1] if needed
        if X_train.dtype != np.float32:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0
        
        # Normalize with CIFAR-10 statistics
        mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
        
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        
        print(f"After preprocessing - X_train: {X_train.shape}, X_test: {X_test.shape}")
        return X_train, X_test
    
    def visualize_samples(self, X: np.ndarray, y: np.ndarray, num_samples: int = 10):
        """Visualize sample images"""
        print(f"Visualizing {num_samples} samples...")
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        # Convert from NCHW to NHWC for visualization
        X_viz = X.transpose(0, 2, 3, 1)
        
        # Denormalize for visualization
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        X_denorm = X_viz * std + mean
        X_denorm = np.clip(X_denorm, 0, 1)
        
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        for i, idx in enumerate(indices):
            img = X_denorm[idx]
            axes[i].imshow(img)
            axes[i].set_title(f'{self.classes[y[idx]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        print("Sample visualization completed!")

# Rest of the classes remain the same as before...
class PatchEmbedding:
    """Turns a 2D input image into a 1D sequence of patch embeddings"""
    
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3, embedding_dim: int = 128):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Manual convolution equivalent using linear projection
        patch_dim = in_channels * patch_size * patch_size
        self.proj_weight = np.random.randn(patch_dim, embedding_dim) * 0.02
        self.proj_bias = np.zeros(embedding_dim)
    
    def extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Extract patches from input images manually"""
        batch_size, channels, height, width = x.shape
        patches = []
        
        for i in range(0, height, self.patch_size):
            for j in range(0, width, self.patch_size):
                patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                patch_flat = patch.reshape(batch_size, -1)
                patches.append(patch_flat)
        
        return np.stack(patches, axis=1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass - convert image to patch embeddings"""
        batch_size = x.shape[0]
        
        # Extract patches
        patches = self.extract_patches(x)  # (batch_size, num_patches, patch_dim)
        
        # Linear projection
        patches_flat = patches.reshape(-1, patches.shape[-1])
        embeddings_flat = np.dot(patches_flat, self.proj_weight) + self.proj_bias
        embeddings = embeddings_flat.reshape(batch_size, self.num_patches, self.embedding_dim)
        
        return embeddings

class MultiHeadSelfAttention:
    """Multi-head self-attention mechanism using NumPy"""
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 4, dropout: float = 0.0):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Query, Key, Value projections
        self.W_q = np.random.randn(embedding_dim, embedding_dim) * 0.02
        self.W_k = np.random.randn(embedding_dim, embedding_dim) * 0.02
        self.W_v = np.random.randn(embedding_dim, embedding_dim) * 0.02
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * 0.02
        
        self.dropout = dropout
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Stable softmax implementation"""
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections
        Q = np.dot(x, self.W_q)
        K = np.dot(x, self.W_k)
        V = np.dot(x, self.W_v)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn_scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn_weights = self.softmax(attn_scores, axis=-1)
        
        # Apply attention to values
        attn_output = np.matmul(attn_weights, V)
        
        # Concatenate heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = np.dot(attn_output, self.W_o)
        
        return output

class LayerNorm:
    """Layer normalization implementation"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class MLPBlock:
    """Multi-layer perceptron block with GELU activation"""
    
    def __init__(self, embedding_dim: int = 128, mlp_size: int = 512, dropout: float = 0.1):
        self.fc1_weight = np.random.randn(embedding_dim, mlp_size) * 0.02
        self.fc1_bias = np.zeros(mlp_size)
        self.fc2_weight = np.random.randn(mlp_size, embedding_dim) * 0.02
        self.fc2_bias = np.zeros(embedding_dim)
        self.dropout = dropout
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function"""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # First linear layer + GELU
        x = np.dot(x, self.fc1_weight) + self.fc1_bias
        x = self.gelu(x)
        
        # Second linear layer
        x = np.dot(x, self.fc2_weight) + self.fc2_bias
        
        return x

class TransformerEncoderBlock:
    """Transformer encoder block with residual connections"""
    
    def __init__(self, embedding_dim: int = 128, num_heads: int = 4, 
                 mlp_size: int = 512, dropout: float = 0.1):
        self.norm1 = LayerNorm(embedding_dim)
        self.attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.norm2 = LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_size, dropout)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Self-attention with residual connection
        residual = x
        x = self.norm1.forward(x)
        x = self.attention.forward(x)
        x = residual + x
        
        # MLP with residual connection
        residual = x
        x = self.norm2.forward(x)
        x = self.mlp.forward(x)
        x = residual + x
        
        return x

class VisionTransformer:
    """Complete Vision Transformer model using NumPy"""
    
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 num_classes: int = 10, embedding_dim: int = 128, num_layers: int = 4,
                 num_heads: int = 4, mlp_size: int = 512, dropout: float = 0.1):
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embedding_dim)
        self.num_patches = self.patch_embedding.num_patches
        
        # Class token (learnable)
        self.class_token = np.random.randn(1, 1, embedding_dim) * 0.02
        
        # Positional embeddings (learnable)
        self.position_embedding = np.random.randn(1, self.num_patches + 1, embedding_dim) * 0.02
        
        # Transformer encoder layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                TransformerEncoderBlock(embedding_dim, num_heads, mlp_size, dropout)
            )
        
        # Layer norm and classifier
        self.layer_norm = LayerNorm(embedding_dim)
        self.classifier_weight = np.random.randn(embedding_dim, num_classes) * 0.02
        self.classifier_bias = np.zeros(num_classes)
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax for classification"""
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding.forward(x)  # (batch_size, num_patches, embedding_dim)
        
        # Add class token
        class_tokens = np.repeat(self.class_token, batch_size, axis=0)
        x = np.concatenate([class_tokens, x], axis=1)
        
        # Add positional embeddings
        x = x + self.position_embedding
        
        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer.forward(x)
        
        # Classification head (use class token)
        x = self.layer_norm.forward(x)
        class_token_output = x[:, 0]  # (batch_size, embedding_dim)
        
        # Final classification
        logits = np.dot(class_token_output, self.classifier_weight) + self.classifier_bias
        
        return logits

class Trainer:
    """Training utilities using NumPy"""
    
    def __init__(self, model: VisionTransformer, learning_rate: float = 0.001):
        self.model = model
        self.learning_rate = learning_rate
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def cross_entropy_loss(self, logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cross-entropy loss and gradients"""
        batch_size = logits.shape[0]
        
        # Softmax probabilities
        probs = self.softmax_stable(logits)
        
        # Cross-entropy loss
        loss = -np.log(probs[np.arange(batch_size), labels] + 1e-8).mean()
        
        # Gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), labels] -= 1
        grad_logits /= batch_size
        
        return loss, grad_logits
    
    def softmax_stable(self, x: np.ndarray) -> np.ndarray:
        """Stable softmax"""
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)
    
    def compute_accuracy(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute accuracy"""
        predictions = np.argmax(logits, axis=1)
        return np.mean(predictions == labels)
    
    def compute_classifier_gradient(self, class_token_output: np.ndarray, grad_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for classifier weights and biases"""
        # Gradient for classifier weights
        grad_weights = np.dot(class_token_output.T, grad_logits)
        # Gradient for classifier biases
        grad_biases = np.sum(grad_logits, axis=0)
        return grad_weights, grad_biases
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        """Train for one epoch"""
        num_samples = X_train.shape[0]
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            
            if len(X_batch) == 0:
                continue
                
            # Forward pass
            logits = self.model.forward(X_batch)
            
            # Compute loss and accuracy
            loss, grad_logits = self.cross_entropy_loss(logits, y_batch)
            accuracy = self.compute_accuracy(logits, y_batch)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1
            
            # Get class token output for gradient computation
            batch_size_current = X_batch.shape[0]
            class_tokens = np.repeat(self.model.class_token, batch_size_current, axis=0)
            patches = self.model.patch_embedding.forward(X_batch)
            x_with_token = np.concatenate([class_tokens, patches], axis=1)
            x_with_pos = x_with_token + self.model.position_embedding
            
            # Pass through encoder layers to get class token output
            for layer in self.model.encoder_layers:
                x_with_pos = layer.forward(x_with_pos)
            
            class_token_output = self.model.layer_norm.forward(x_with_pos)[:, 0]
            
            # Compute gradients for classifier
            grad_weights, grad_biases = self.compute_classifier_gradient(class_token_output, grad_logits)
            
            # Update classifier weights and biases
            self.model.classifier_weight -= self.learning_rate * grad_weights
            self.model.classifier_bias -= self.learning_rate * grad_biases
        
        return epoch_loss / num_batches, epoch_accuracy / num_batches
    
    def validate(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int = 32) -> Tuple[float, float]:
        """Validate the model"""
        num_samples = X_val.shape[0]
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            X_batch = X_val[i:end_idx]
            y_batch = y_val[i:end_idx]
            
            if len(X_batch) == 0:
                continue
                
            logits = self.model.forward(X_batch)
            loss, _ = self.cross_entropy_loss(logits, y_batch)
            accuracy = self.compute_accuracy(logits, y_batch)
            
            val_loss += loss
            val_accuracy += accuracy
            num_batches += 1
        
        return val_loss / num_batches, val_accuracy / num_batches
    
    def plot_training_history(self):
        """Plot training history"""
        print("Plotting training history...")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        plt.plot(self.val_losses, label='Val Loss', marker='s')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy', marker='o')
        plt.plot(self.val_accuracies, label='Val Accuracy', marker='s')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("Training plots displayed!")

def main():
    """Main training loop"""
    print("=" * 60)
    print("VISION TRANSFORMER WITH SPECIFIC FILE PATH")
    print("=" * 60)
    
    try:
        # Load ACTUAL data from the specific path
        loader = CIFAR10Loader()
        data_path = "/home/yagati/Downloads/cifar-10"
        
        print(f"Using data path: {data_path}")
        X_train, y_train, X_test, y_test = loader.load_all_data(data_path)
        X_train, X_test = loader.preprocess_data(X_train, X_test)
        
        # Visualize samples from ACTUAL data
        print("\nVisualizing training samples...")
        loader.visualize_samples(X_train, y_train)
        
        # Create model
        print("Initializing Vision Transformer...")
        model = VisionTransformer(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embedding_dim=128,
            num_layers=4,
            num_heads=4,
            mlp_size=512
        )
        
        # Test forward pass
        print("\nTesting forward pass...")
        test_batch = X_train[:2]
        logits = model.forward(test_batch)
        print(f"Input shape: {test_batch.shape}")
        print(f"Output logits shape: {logits.shape}")
        
        # Create trainer
        trainer = Trainer(model, learning_rate=0.001)
        
        # Use smaller dataset for faster training (you can increase these numbers)
        train_samples = min(2000, len(X_train))
        test_samples = min(500, len(X_test))
        
        X_tr = X_train[:train_samples]
        y_tr = y_train[:train_samples]
        X_val = X_test[:test_samples]
        y_val = y_test[:test_samples]
        
        print(f"\nTraining samples: {len(X_tr)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Training for 10 epochs
        print("\nStarting training for 10 epochs...")
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
        print("-" * 50)
        
        for epoch in range(10):
            # Train one epoch
            train_loss, train_acc = trainer.train_epoch(X_tr, y_tr, batch_size=32)
            
            # Validate
            val_loss, val_acc = trainer.validate(X_val, y_val, batch_size=32)
            
            # Store history
            trainer.train_losses.append(train_loss)
            trainer.train_accuracies.append(train_acc)
            trainer.val_losses.append(val_loss)
            trainer.val_accuracies.append(val_acc)
            
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.4f} | {val_loss:8.4f} | {val_acc:7.4f}")
        
        # Plot training history
        trainer.plot_training_history()
        
        # Show some predictions
        print("\nShowing some predictions...")
        num_test_samples = min(10, len(X_test))
        test_indices = np.random.choice(len(X_test), num_test_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(test_indices):
            # Get single sample
            X_sample = X_test[idx:idx+1]
            y_true = y_test[idx]
            
            # Predict
            logits = model.forward(X_sample)
            pred = np.argmax(logits, axis=1)[0]
            
            # Denormalize for visualization
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
            std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
            img_denorm = X_sample[0] * std[0] + mean[0]
            img_denorm = np.clip(img_denorm, 0, 1)
            img = img_denorm.transpose(1, 2, 0)  # Convert back to HWC for display
            
            axes[i].imshow(img)
            color = 'green' if pred == y_true else 'red'
            axes[i].set_title(f'True: {loader.classes[y_true]}\nPred: {loader.classes[pred]}', color=color)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
