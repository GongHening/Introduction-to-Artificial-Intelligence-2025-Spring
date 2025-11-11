from autograd.BaseGraph import Graph
from autograd.BaseNode import *
import numpy as np
from mnist import *
from scipy.ndimage import rotate, shift, gaussian_filter
import pickle
import os
from util import setseed

setseed(0)
os.makedirs("model", exist_ok=True)

def augment(batch_X):
    """Enhanced augmentation with rotation, translation, and multiple noise types"""
    X_aug = []
    for x in batch_X:
        img = x.reshape(28, 28)
        
        # Apply geometric transformations
        img = rotate(img, angle=np.random.uniform(-20, 20), reshape=False, mode='nearest')
        img = shift(img, shift=(np.random.uniform(-2.5, 2.5), np.random.uniform(-2.5, 2.5)), mode='nearest')
        
        # Add multiple types of noise
        noise_type = np.random.choice(['gaussian', 'salt_pepper', 'speckle'])
        
        if noise_type == 'gaussian':
            # Gaussian noise
            img += np.random.normal(0, 0.07, img.shape)
        elif noise_type == 'salt_pepper':
            # Salt-and-pepper noise
            salt = np.random.rand(*img.shape) < 0.02
            pepper = np.random.rand(*img.shape) < 0.02
            img[salt] = 1.0
            img[pepper] = 0.0
        else:  # speckle
            # Speckle noise (multiplicative)
            img = img * (1 + np.random.normal(0, 0.1, img.shape))
        
        # Apply slight blur occasionally
        if np.random.rand() < 0.3:
            img = gaussian_filter(img, sigma=0.5)
            
        X_aug.append(np.clip(img.flatten(), 0, 1))
    return np.stack(X_aug)

def train_model():
    X_train, y_train = trn_X, trn_Y
    graph=Graph([
        Linear(num_feat, 256),
        BatchNorm(256), relu(), Dropout(0.2),
        Linear(256, 128),
        Linear(128, 10),
        LogSoftmax(),
        NLLLoss(y_train)
    ])

    lr = 1e-3
    wd1 = 1e-5  # L1 regularization
    wd2 = 1e-4
    batch_size = 128
    epochs = 50
    best_acc = 0

    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        num_batches = int(np.ceil(len(X_train) / batch_size))
        total_loss, correct = 0, 0

        for i in range(num_batches):
            batch_idx = indices[i * batch_size:(i + 1) * batch_size]
            batch_X = X_train[batch_idx]
            batch_Y = y_train[batch_idx]
            batch_X = augment(batch_X)

            graph.flush()
            graph[-1].y = batch_Y
            pred, loss = graph.forward(batch_X)[-2:]
            graph.backward()
            graph.optimstep(lr, wd1, wd2)

            total_loss += loss
            correct += np.sum(np.argmax(pred, axis=1) == batch_Y)

        train_acc = correct / len(X_train)

        # Validation accuracy
        graph.eval()
        graph.flush()
        val_pred = graph.forward(val_X, removelossnode=True)[-1]
        val_acc = np.mean(np.argmax(val_pred, axis=1) == val_Y)

        if val_acc > best_acc:
            best_acc = val_acc
            with open("model/Your.model", "wb") as f:
                pickle.dump(graph, f)

        print(f"Epoch {epoch+1}: Train Loss={total_loss/num_batches:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

if __name__ == "__main__":
    train_model()