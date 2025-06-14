import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import joblib
from ..config import Config


class LiquidityPredictionModel(nn.Module):
    """Neural network for predicting liquidity metrics"""
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DeFiMLEngine:
    """Machine learning engine for DeFi predictions"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Config.MODEL_STORAGE_PATH / 'liquidity_model.pth'
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.scaler_path = self.model_path.with_suffix('.scaler')
        
        self.model: Optional[LiquidityPredictionModel] = None
        self.scaler: StandardScaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_history: Dict[str, list] = {'loss': [], 'val_loss': []}
        
        self.load_or_initialize_model()
    
    def load_or_initialize_model(self, input_size: int = 10) -> None:
        """Load existing model or create new one"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = checkpoint['model']
            self.training_history = checkpoint.get('history', {'loss': [], 'val_loss': []})
            
            # Load scaler if exists
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                
            self.model.to(self.device)
            self.model.eval()
        except (FileNotFoundError, KeyError):
            self.model = LiquidityPredictionModel(input_size=input_size)
            self.model.to(self.device)
            self.save_model()
    
    def save_model(self) -> None:
        """Save model and scaler to disk"""
        checkpoint = {
            'model': self.model,
            'history': self.training_history,
            'input_size': self.model.network[0].in_features
        }
        torch.save(checkpoint, self.model_path)
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, Any]:
        """Train the model on provided data
        
        Args:
            X: Input features
            y: Target values
            validation_split: Fraction of data for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Training metrics
        """
        # Split data
        val_size = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        # Normalize inputs
        X_train = self.scaler.fit_transform(X[train_indices])
        X_val = self.scaler.transform(X[val_indices])
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y[train_indices]).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y[val_indices]).unsqueeze(1).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        self.model.train()
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()
            
            # Record history
            self.training_history['loss'].append(epoch_loss / len(X_train_tensor))
            self.training_history['val_loss'].append(val_loss)
            
            self.model.train()
        
        return {
            'final_loss': self.training_history['loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        self.model.eval()
        
        # Normalize inputs
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.squeeze()