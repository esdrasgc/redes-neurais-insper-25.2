import numpy as np

# ---------- Activation Functions ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # z shape: (m, C)
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / (np.sum(exp_z, axis=1, keepdims=True) + 1e-8)

# ---------- Loss Functions ----------
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) + 
                   (1 - y_true) * np.log(1 - y_pred + 1e-8))

def bce_derivative(y_true, y_pred):
    return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-8)

def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


# ---------- MLP Class ----------
class MLPClassifier:
    def __init__(self, layer_sizes, lr=0.01, epochs=1000, scale=0.1, verbose=True, seed=None):
        """
        layer_sizes: list com número de neurônios [input, hidden1, ..., output]
        lr: learning rate
        epochs: número de épocas de treinamento
        scale: fator de escala para inicialização dos pesos
        verbose: se True, printa loss a cada 100 épocas
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.epochs = epochs
        self.scale = scale
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)
        self.theta = self._init_weights(layer_sizes, scale)
        self.history_ = {}

    def _init_weights(self, layer_sizes, scale):
        weights = []
        for k in range(len(layer_sizes)-1):
            U_k = layer_sizes[k]
            U_k_plus_1 = layer_sizes[k+1]
            weight = scale * np.random.randn(U_k+1, U_k_plus_1)
            weights.append(weight)
        return [weights[:-1], weights[-1]]

    def _forward(self, X):
        a = X.T 
        activations = [a]
        zs = []
        
        for W in self.theta[0]:
            z = W[0] + np.dot(a.T, W[1:]) 
            a = relu(z).T 
            zs.append(z)
            activations.append(a)

        z_out = self.theta[1][0] + np.dot(a.T, self.theta[1][1:]) 
        out_dim = z_out.shape[1]
        if out_dim == 1:
            y_pred = sigmoid(z_out) 
        else:
            y_pred = softmax(z_out)
        zs.append(z_out)
        return y_pred, zs, activations

    def fit(self, X, y, track_loss=False, X_val=None, y_val=None):
        # inicializa histórico
        self.history_ = {}
        if track_loss:
            self.history_["loss"] = []
            if X_val is not None and y_val is not None:
                self.history_["val_loss"] = []
        m = X.shape[0]
        out_dim = self.layer_sizes[-1]
        y_proc = y
        if y_proc.ndim == 1:
            y_proc = y_proc.reshape(-1, 1)
        if out_dim > 1:
            if y_proc.shape[1] == 1 and np.issubdtype(y_proc.dtype, np.integer):
                one_hot = np.zeros((m, out_dim))
                one_hot[np.arange(m), y_proc.flatten()] = 1
                y_proc = one_hot
        else:
            if y_proc.shape[1] != 1:
                y_proc = y_proc[:, :1]

        for epoch in range(self.epochs):
            # -------- Forward pass --------
            y_pred, zs, activations = self._forward(X)

            # Loss
            if y_pred.shape[1] == 1:
                loss = binary_cross_entropy(y_proc, y_pred)
            else:
                loss = categorical_cross_entropy(y_proc, y_pred)
            if track_loss:
                self.history_["loss"].append(float(loss))
                # validação opcional
                if X_val is not None and y_val is not None:
                    yv = y_val
                    if yv.ndim == 1:
                        yv = yv.reshape(-1, 1)
                    if out_dim > 1:
                        if yv.shape[1] == 1 and np.issubdtype(yv.dtype, np.integer):
                            one_hot_v = np.zeros((yv.shape[0], out_dim))
                            one_hot_v[np.arange(yv.shape[0]), yv.flatten()] = 1
                            yv = one_hot_v
                    y_pred_val, _, _ = self._forward(X_val)
                    if y_pred_val.shape[1] == 1:
                        vloss = binary_cross_entropy(yv, y_pred_val)
                    else:
                        vloss = categorical_cross_entropy(yv, y_pred_val)
                    self.history_["val_loss"].append(float(vloss))

            # -------- Backpropagation --------
            delta = (y_pred - y_proc)  # (m, out)

            # Gradiente da camada de saída
            grad_W_out = np.vstack([
                np.ones((1, activations[-1].shape[1])),  # bias
                activations[-1]
            ]) @ delta / X.shape[0]

            self.theta[1] -= self.lr * grad_W_out

            for l in range(len(self.theta[0]) - 1, -1, -1):
                z = zs[l]
                a_prev = activations[l]
                if l == len(self.theta[0]) - 1:
                    W_next_no_bias = self.theta[1][1:, :]
                else:
                    W_next_no_bias = self.theta[0][l+1][1:, :]

                delta = (delta @ W_next_no_bias.T) * relu_derivative(z)

                grad_W = np.vstack([
                    np.ones((1, a_prev.shape[1])),  # bias
                    a_prev
                ]) @ delta / X.shape[0]

                self.theta[0][l] -= self.lr * grad_W

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss = {loss:.4f}")



    def predict_proba(self, X):
        y_pred, _, _ = self._forward(X)
        return y_pred 

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        if proba.shape[1] == 1:
            return (proba > threshold).astype(int)
        return np.argmax(proba, axis=1).reshape(-1, 1)
